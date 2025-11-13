from zenml import step
from pathlib import Path
import yaml

@step
def ingest_data(data_dir: str = "dataset"):
    data_path = Path(data_dir) / "data.yaml"
    with open(data_path, "r") as f:
        data_config = yaml.safe_load(f)

    print(f"Loaded dataset config from: {data_path}")
    print(f"Classes: {data_config.get('names', [])}")
    return str(data_path)
