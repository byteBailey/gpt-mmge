import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


DATASET_DIR = Path(__file__).resolve().parents[3] / "datasets" / "Grocery"
GRAPH_PATH = DATASET_DIR / "Grocery_Aug_non_structure_graph_data.pt"
SPLIT_PATH = DATASET_DIR / "Grocery_split.json"


candidates = {
    "Grocery": [
        "Dried Beans, Grains & Rice",
        "Canned, Jarred & Packaged Foods",
        "Pasta & Noodles",
        "Food & Beverage Gifts",
        "Candy & Chocolate",
        "Condiments & Salad Dressings",
        "Produce",
        "Sauces, Gravies & Marinades",
        "Dairy, Cheese & Eggs",
        "Beverages",
        "Soups, Stocks & Broths",
        "Frozen",
        "Herbs, Spices & Seasonings",
        "Fresh Flowers & Live Indoor Plants",
        "Cooking & Baking",
        "Breads & Bakery",
        "Meat & Seafood",
        "Jams, Jellies & Sweet Spreads",
        "Snack Foods",
        "Breakfast Foods",
    ]
}


class GroceryAugDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        self.synthetic_summary = self.graph.synthetic_summarys

        self.title_synthetic_summary = []
        for text, summary in zip(self.text, self.synthetic_summary):
            if "Title: " in text and "Description: " in text:
                title_part, _ = text.split("; Description: ")
                title = title_part.replace("Title: ", "").strip()
                description = summary.strip()
                new_text = f"Title: {title}; Description: {description}"
                self.title_synthetic_summary.append(new_text)

        candidates_str = ", ".join(candidates["Grocery"])
        self.prompt = (
            "Which category does the product seem to belong to? "
            f"Choose from the following options: {candidates_str}.\n\nAnswer:"
        )
        self.graph_type = "Text Attributed Graph"

        self.num_features = 768
        self.num_classes = 20
        print(f"label mapping: {self.graph.label_texts}")

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):
            return {
                "id": index,
                "label": self.graph.label_texts[int(self.graph.y[index])],
                "desc": self.title_synthetic_summary[index],
                "question": self.prompt,
            }
        return None

    @property
    def processed_file_names(self):
        return [str(GRAPH_PATH)]

    def get_idx_split(self):
        with open(SPLIT_PATH, "r", encoding="utf-8") as file:
            loaded_data_dict = json.load(file)

        train_ids = [int(i) for i in loaded_data_dict["train"]]
        val_ids = [int(i) for i in loaded_data_dict["val"]]
        test_ids = [int(i) for i in loaded_data_dict["test"]]

        print(
            f"Loaded data from {SPLIT_PATH}: train_id length = {len(train_ids)}, "
            f"test_id length = {len(test_ids)}, val_id length = {len(val_ids)}"
        )

        return {"train": train_ids, "test": test_ids, "val": val_ids}


if __name__ == "__main__":
    dataset = GroceryAugDataset()

    print(dataset.graph)
    print(dataset.prompt)
    print(json.dumps(dataset[0], indent=4))

    split_ids = dataset.get_idx_split()
    for key, value in split_ids.items():
        print(f"# {key}: {len(value)}")
