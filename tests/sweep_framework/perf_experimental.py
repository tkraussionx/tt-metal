import argparse
import sqlite3
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

VECTOR_TABLE_SUFFIX = "_vector"
RESULT_TABLE_SUFFIX = "_result"


def broadcast_to_color(broadcast):
    if broadcast == "hw":
        return "r"
    elif broadcast == "h":
        return "g"
    elif broadcast == "w":
        return "b"
    else:
        return "m"


def broadcast_to_index(broadcast):
    if broadcast == "hw":
        return 3
    elif broadcast == "h":
        return 2
    elif broadcast == "w":
        return 1
    else:
        return 0


# def broadcast_to_data(broadcast, data):
#     if broadcast == "hw":
#         return data["output_tiles"]
#     elif broadcast == "h":
#         return data["input_w"]
#     elif broadcast == "w":
#         return data["input_h"]
#     else:
#         return data["output_tiles"]


def broadcast_to_data(broadcast, data):
    if broadcast == "hw":
        return data["output_tiles"]
    elif broadcast == "h":  # input_h = 1 and broadcasts to input_h
        return data["input_w"]
        # return data["output_tiles"]
    elif broadcast == "w":
        return data["input_h"]
        # return data["output_tiles"]
    else:
        return data["output_tiles"]


def read_perf_raw_data(file):
    perf_raw_data = {}

    # Open the SQLite database
    conn = sqlite3.connect(file)
    cursor = conn.cursor()

    # Execute a query to fetch all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")

    # Fetch all table names and print them
    tables = cursor.fetchall()
    for table in tables:
        print(table[0])

    table_prefix = file.split("/")[-1].split(".")[0]
    results_table = table_prefix + RESULT_TABLE_SUFFIX
    vectors_table = table_prefix + VECTOR_TABLE_SUFFIX

    # print(f"SELECT * FROM {results_table}")
    # # Execute a query to fetch all rows from a table
    # cursor.execute(f"SELECT * FROM {results_table} WHERE status = 'TestStatus.PASS'")

    # # Fetch all PASS runs, and print them.
    # rows = cursor.fetchall()
    # for row in rows:
    #     print(row)

    # for each passed row find the corresponding vector
    cursor.execute(
        f"SELECT timestamp, suite_name, vector_id, device_perf FROM {results_table} WHERE status = 'TestStatus.PASS' and device_perf != ''"
    )
    # Fetch all rows and print them
    rows = cursor.fetchall()
    for row in rows:
        if row[3] == "":  # no data!
            print("{vector_id} has no perf data")
            continue
        timestamp = row[0]
        suite_name = row[1]
        vector_id = row[2]
        # device_perf = dict(json.loads(row[2]))
        device_perf = json.loads(row[3].replace("'", '"'))
        print(timestamp)
        cursor.execute(
            f"SELECT batch_sizes, height, width, broadcast FROM {vectors_table} WHERE vector_id = '{vector_id}'"
        )
        vector = cursor.fetchone()
        print(vector)
        if vector is None:
            print(f"Vector {vector_id} not found")
            continue

        print(device_perf["DEVICE FW DURATION [ns]"])

        if suite_name not in perf_raw_data:
            perf_raw_data[suite_name] = {}

        perf_raw_data[suite_name][vector_id] = {
            "timestamp": timestamp,
            "broadcast": vector[3],
            "input_vector": vector[:-1],
            "device_perf": device_perf,
        }

    # Close the database connection
    conn.close()

    return perf_raw_data


def process_raw_perf(perf_raw_data):
    perf_data = {}

    # X = [] # input_data larger if broadcast
    # Y = [] # device_perf DEVICE FW DURATION [ns]
    # labels = [] # input_vector broadcast

    for suite_name, suite_data in perf_raw_data.items():
        if suite_name not in perf_data:
            perf_data[suite_name] = {}

        for vector_id, data in suite_data.items():
            if data["broadcast"] not in perf_data[suite_name]:
                perf_data[suite_name][data["broadcast"]] = {}
            # ignoring batch size for now as it is composite op and doesnt get perf evaluated anyway

            if "output_tiles" not in perf_data[suite_name][data["broadcast"]]:
                perf_data[suite_name][data["broadcast"]]["output_tiles"] = []
            perf_data[suite_name][data["broadcast"]]["output_tiles"].append(
                int(data["input_vector"][1]) * int(data["input_vector"][2]) / 32 / 32
            )

            if "input_h" not in perf_data[suite_name][data["broadcast"]]:
                perf_data[suite_name][data["broadcast"]]["input_h"] = []
            perf_data[suite_name][data["broadcast"]]["input_h"].append(int(data["input_vector"][1]))

            if "input_w" not in perf_data[suite_name][data["broadcast"]]:
                perf_data[suite_name][data["broadcast"]]["input_w"] = []
            perf_data[suite_name][data["broadcast"]]["input_w"].append(int(data["input_vector"][2]))

            if "device_perf" not in perf_data[suite_name][data["broadcast"]]:
                perf_data[suite_name][data["broadcast"]]["device_perf"] = []
            perf_data[suite_name][data["broadcast"]]["device_perf"].append(
                int(data["device_perf"]["DEVICE FW DURATION [ns]"])
            )

    return perf_data


def plot_perf(perf_data):
    # Create a new figure
    fig, axes = plt.subplots(ncols=4, nrows=len(perf_data.keys()), figsize=(16, 8), dpi=120)

    # Plotting output_tiles over device_perf with broadcast labels
    subplot_idx = 0
    for suite_name, suite_data in perf_data.items():
        ax = axes[subplot_idx]
        subplot_idx += 1
        for broadcast, data in suite_data.items():
            ax[broadcast_to_index(broadcast)].scatter(
                broadcast_to_data(broadcast, data),
                data["device_perf"],
                label=broadcast,
                c=broadcast_to_color(broadcast),
            )

        ax[broadcast_to_index(broadcast)].set_xlabel("Input Tiles")
        ax[broadcast_to_index(broadcast)].set_ylabel("(DEVICE FW DURATION [ns])")
        ax[broadcast_to_index(broadcast)].legend()

    fig.savefig("perf_plot.png")

    ##
    # fig, axes = plt.subplots(ncols=4, nrows=len(perf_data.keys()), figsize=(16, 8), dpi=120)

    # subplot_idx = 0
    # for suite_name, suite_data in perf_data.items():
    #     ax = axes[subplot_idx]
    #     subplot_idx += 1
    #     for broadcast, data in suite_data.items():

    #         # Perform PCA on the data
    #         pca = PCA(n_components=2)
    #         xx = np.array([data["output_tiles"], data["input_w"], data["input_h"]]).T

    #         # Standardize the data
    #         scaler = StandardScaler()
    #         xx_scaled = scaler.fit_transform(xx)

    #         # Perform PCA on the data
    #         pca = PCA(n_components=2)
    #         pca_data = pca.fit_transform(xx_scaled)

    #         # Plot the two highest coefficients
    #         ax[broadcast_to_index(broadcast)].scatter(pca_data[:, 0], pca_data[:, 1], label=broadcast, c=broadcast_to_color(broadcast))

    #         # ax[broadcast_to_index(broadcast)].scatter(broadcast_to_data(
    #         #     broadcast, data),
    #         #     data["device_perf"],
    #         #     label=broadcast,
    #         #     c=broadcast_to_color(broadcast))

    #         ax[broadcast_to_index(broadcast)].set_xlabel('pca[0]')
    #         ax[broadcast_to_index(broadcast)].set_ylabel('pca[1]')
    #         ax[broadcast_to_index(broadcast)].legend()
    # fig.savefig('perf_plot_pca.png')

    ##

    # 3d plot for funky stuff
    for suite_name, suite_data in perf_data.items():
        fig = plt.figure()
        # ax = fig.gca(projection='3d')
        ax = fig.add_subplot(projection="3d")

        for broadcast, data in suite_data.items():
            if broadcast != "h":
                continue

            ax.scatter(
                data["output_tiles"],
                data["input_w"],
                zs=data["device_perf"],
                zdir="y",
                label=broadcast,
                c=broadcast_to_color(broadcast),
            )

        # Make legend, set axes limits and labels
        ax.legend()
        ax.set_xlim(0, max(data["output_tiles"]))
        ax.set_ylim(0, max(data["input_w"]))
        ax.set_zlim(0, max(data["device_perf"]))
        ax.set_xlabel("output_tiles")
        ax.set_ylabel("input_w")
        ax.set_zlabel("device_perf")

        fig.savefig(f"perf_plot_{suite_name}.png")


def main():
    parser = argparse.ArgumentParser(description="Print contents of a .sqlite file")
    parser.add_argument("-f", "--file", type=str, help="Path to the .sqlite file")

    args = parser.parse_args()

    perf_raw_data = read_perf_raw_data(args.file)
    perf_data = process_raw_perf(perf_raw_data)

    plot_perf(perf_data)

    # input_shape_a = (*batch_sizes, height, width)
    # input_shape_b = (*batch_sizes, height, width)
    # if broadcast == "hw":
    #     input_shape_b = (*batch_sizes, 1, 1)
    # elif broadcast == "h":
    #     input_shape_b = (*batch_sizes, 1, width)
    # elif broadcast == "w":
    #     input_shape_b = (*batch_sizes, height, 1)


if __name__ == "__main__":
    main()
