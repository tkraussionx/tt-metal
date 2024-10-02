import requests
import json
from elasticsearch import Elasticsearch
import os

d = os.getcwd()
os.chdir("tests/sweep_framework")
import elastic_config

os.chdir(d)

# Get Elasticsearch connection details from environment or use defaults
ELASTIC_CONNECTION_STRING = elastic_config.get_elastic_url("cloud")
ELASTIC_USERNAME = os.getenv("ELASTIC_USERNAME", "elastic")
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD", "changeme")

# Initialize Elasticsearch client
es_client = Elasticsearch(ELASTIC_CONNECTION_STRING, basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD))


def sanitize_inputs(test_vectors):
    info_field_names = ["sweep_name", "suite_name", "vector_id", "input_hash"]
    header_info = []
    for vector in test_vectors:
        header = dict()
        for field in info_field_names:
            header[field] = vector.pop(field)
        vector.pop("timestamp")
        vector.pop("tag")
        header_info.append(header)
    return header_info, test_vectors


def get_suite_vectors(client, suite, tag, module_name):
    response = es_client.search(
        index=f"ttnn_sweeps_test_vectors_{module_name}",
        query={
            "bool": {
                "must": [
                    {"match": {"status": "VectorStatus.CURRENT"}},
                    {"match": {"suite_name.keyword": suite}},
                    {"match": {"tag.keyword": tag}},
                    {"match": {"sweep_name": module_name}},
                ]
            }
        },
        size=10000,
    )
    test_ids = [hit["_id"] for hit in response["hits"]["hits"]]
    test_vectors = [hit["_source"] for hit in response["hits"]["hits"]]
    for i in range(len(test_ids)):
        test_vectors[i]["vector_id"] = test_ids[i]
    header_info, test_vectors = sanitize_inputs(test_vectors)
    return header_info, test_vectors


def serialize_result(result):
    serialized = {}
    for key, value in result["_source"].items():
        if key == "status":
            serialized[key] = TestStatus(value).name
        elif key == "validity":
            serialized[key] = VectorValidity(value).name
        else:
            serialized[key] = value
    return serialized


def get_sweep_results(tag, module_name, time_elapsed=15):
    results = es_client.search(
        index=f"ttnn_sweeps_test_results_{module_name}",
        query={
            "bool": {
                "must": [{"match": {"sweep_name": module_name}}],
                "filter": {"range": {"timestamp": {"gte": f"now-{time_elapsed}m", "lte": "now"}}},
            }
        },
        size=10000,
    )
    return results["hits"]["hits"]


# Main execution
tag = "jkruer-copy"
module_name = "copy.copy"
idpacket_vectors = get_suite_vectors(es_client, "nightly", tag, module_name)
# results = get_sweep_results(tag, module_name)


def associate_vectors_with_results(idpacket_vectors, results):
    idpkt_vectors = zip(idpacket_vectors[0], idpacket_vectors[1])
    vector_id_to_vector = {idpkt["vector_id"]: vector for (idpkt, vector) in idpkt_vectors}
    for result in results:
        vector_id = result.get("vector_id")
        if not vector_id:
            continue
        result["vector"] = vector_id_to_vector[vector_id]
    return results


results = associate_vectors_with_results(idpacket_vectors, results)
print(results)
