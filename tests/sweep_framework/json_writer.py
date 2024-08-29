import json
import os


class JSONVectorManager:
    def __init__(self, json_file_path):
        """
        Initializes the VectorManager with a JSON file.

        Args:
            json_file_path (str): Path to the JSON file containing vector data.
        """
        self.json_file_path = json_file_path
        self.data = self._load_data()

    def _load_data(self):
        """Load vector data from the provided JSON file. Create the file if it doesn't exist."""
        if not os.path.exists(self.json_file_path):
            # If the file doesn't exist, create it with an initial empty structure
            with open(self.json_file_path, "w") as f:
                json.dump({"old_vector_ids": [], "serialized_vectors": {}}, f, indent=4)

        # Load data from the file
        with open(self.json_file_path, "r") as f:
            return json.load(f)

    def update_vectors(self, old_vector_id):
        """
        Updates the status of the given old vectors to 'ARCHIVED'.

        Args:
            old_vector_ids (list): List of vector IDs to be updated.
        """
        # Update old vectors to 'ARCHIVED' status
        if old_vector_id in self.data.get("serialized_vectors", {}):
            self.data["serialized_vectors"][old_vector_id]["status"] = "ARCHIVED"
        else:
            self.data["serialized_vectors"][old_vector_id] = {"status": "ARCHIVED"}

        self._save_data()

    def index(self, new_vector_id, body):
        """
        Indexes a new vector by adding it to the JSON data.

        Args:
            new_vector_id (str): The ID of the new vector to be indexed.
            body (dict): The body of the new vector data.
        """
        # Index the new vector by adding/updating an entry
        self.data.setdefault("serialized_vectors", {})[new_vector_id] = body

        self._save_data()

    def _save_data(self):
        """Save the modified data back to the JSON file."""
        with open(self.json_file_path, "w") as f:
            json.dump(self.data, f, indent=4)
