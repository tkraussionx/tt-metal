import requests
import json
from elasticsearch import Elasticsearch
import os
import elastic_config
import frozendict

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


def get_suite_vectors(client, module_name):
    response = client.search(
        index=f"ttnn_sweeps_test_vectors_{module_name}",
        query={
            "bool": {
                "must": [
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


def get_sweep_results(module_name, time_elapsed=15):
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
import argparse

parser = argparse.ArgumentParser(description="Get sweep results")
parser.add_argument("--module_name", type=str, required=True, help="Module name")
parser.add_argument("-t", "--time_range", type=int, default=60, help="Time range in minutes (default: 60)")

args = parser.parse_args()
module_name = args.module_name
idpacket_vectors = get_suite_vectors(es_client, module_name)
results = get_sweep_results(module_name, args.time_range)


# flatten each result dict
def flatten_dict(d):
    flattened = {}
    for key, value in d.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flattened[f"{key}_{sub_key}"] = sub_value
        else:
            flattened[key] = value
    return flattened


results = [flatten_dict(result) for result in results]

# remove the _source prefix from each key
results = [{key.replace("_source_", ""): value for key, value in result.items()} for result in results]


def associate_vectors_with_results(idpacket_vectors, results):
    idpkt_vectors = list(zip(idpacket_vectors[0], idpacket_vectors[1]))
    vector_id_to_vector = {idpkt["vector_id"]: vector for (idpkt, vector) in idpkt_vectors}
    ret = []
    for result in results:
        vector_id = result.get("vector_id")
        if not vector_id:
            continue
        copy = result.copy()
        copy["vector"] = vector_id_to_vector.get(vector_id)
        ret.append(copy)
    return ret


results = associate_vectors_with_results(idpacket_vectors, results)
results = [frozendict.deepfreeze(result) for result in results]
results = set(results)


# convert results frozendicts to dicts
def convert_to_dict(fd):
    res = dict(fd)
    for key, value in res.items():
        if isinstance(value, frozendict.frozendict):
            res[key] = convert_to_dict(value)
    return res


results = [convert_to_dict(result) for result in results]


# Group results by their exceptions for grouping exceptions, only use the part
# of the exception up until the backtrace; use this component as the key as
# well. Also compute the percent of the total that each exception group represents.
exception_groups = {}
total_results = len(results)

for result in results:
    exception = result.get("exception", "No Exception")
    # Extract the part of the exception up until the backtrace
    exception_key = exception.split("backtrace:", 1)[0].strip()
    try:
        exception_key = float(exception_key)
        if exception_key <= 1:
            exception_key = f"PCC mismatch"
    except ValueError:
        pass
    if exception_key not in exception_groups:
        exception_groups[exception_key] = []
    exception_groups[exception_key].append(result)

# Print vectors and exceptions for each group
for exception, group in exception_groups.items():
    group_size = len(group)
    percentage = (group_size / total_results) * 100
    print(f"\nException: {exception}")
    print("Sample vectors:")
    for result in group[:3]:  # Print up to 3 vectors per group
        print(f"  {result.get('vector', 'No vector information')}")
    print(f"Total occurrences: {group_size}")
    print(f"Percentage of total: {percentage:.2f}%")
    print("-" * 50)

# create columns names from all keys across all results
columns = set()
for result in results:
    columns.update(result.keys())

# remove "_ignored" from the columns
columns.discard("_ignored")

# remove "_ignored" from every result
for result in results:
    result.pop("_ignored", None)

columns = list(columns)

# put "vector" at the beginning of the columns
columns.insert(0, columns.pop(columns.index("vector")))

# convert results set into a CSV using the columns given by the dictionary keys in the first result
import csv
from datetime import datetime
import os

if results:
    # if any column is not present, just put N/A for that column.
    for result in results:
        for key in columns:
            if key not in result:
                result[key] = "N/A"
    results_dir = os.path.expanduser("~") + "/sweep_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    with open(
        f'{results_dir}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{module_name}_results.csv', "w", newline=""
    ) as output_file:
        dict_writer = csv.DictWriter(output_file, columns)
        dict_writer.writeheader()
        dict_writer.writerows(results)
else:
    print("No results to write to CSV")
