from src.dataset.arts import ArtsDataset
from src.dataset.cd import CDDataset
from src.dataset.grocery import GroceryDataset
from src.dataset.movies import MoviesDataset
from src.dataset.toys import ToysDataset

load_dataset = {
    'arts': ArtsDataset,
    'cd': CDDataset,
    'grocery': GroceryDataset,
    'movies': MoviesDataset,
    'toys': ToysDataset,
}
