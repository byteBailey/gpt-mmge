import json

import torch
from torch.utils.data import Dataset


GRAPH_PATH = "/home/liyijun/gpt-mmge/MAGB/Arts/FineFeatures_convert/Arts_arts_ori_text_aug_imgText_ori_img_graph_data.pt"
SPLIT_PATH = "/home/liyijun/gpt-mmge/MAGB/Arts/FineFeatures_convert/Arts_split.json"


candidates = {
    "Arts": [
        "Beading & Jewelry Making",
        "Crafting",
        "Fabric",
        "Knitting & Crochet",
        "Painting, Drawing & Art Supplies",
        "Sewing",
        "Scrapbooking & Stamping",
    ]
}


class ArtsDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts

        candidates_str = ", ".join(candidates["Arts"])
        self.prompt = (
            "Which category does the product seem to belong to? "
            f"Choose from the following options: {candidates_str}.\n\nAnswer:"
        )
        self.graph_type = "Text Attributed Graph"

        self.num_features = 768 * 2
        self.num_classes = 7
        print(f"label mapping: {self.graph.label_texts}")

        if hasattr(self.graph, "img_features"):
            print(
                f"Multimodal fields found: img_features {self.graph.img_features.shape}, "
                f"txt_features {self.graph.txt_features.shape}, "
                f"edge_type {self.graph.edge_type.shape}"
            )
        else:
            print("Multimodal fields not found, will use legacy GAT encoder")

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):
            return {
                "id": index,
                "label": self.graph.label_texts[int(self.graph.y[index])],
                "desc": self.text[index],
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
    dataset = ArtsDataset()

    print(dataset.graph)
    print(dataset.prompt)
    print(json.dumps(dataset[0], indent=4))

    split_ids = dataset.get_idx_split()
    for key, value in split_ids.items():
        print(f"# {key}: {len(value)}")
