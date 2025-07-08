import yaml

def load_config(path="/mnt/g/Authenta/data-generations/Remaskable/Remaskable/config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)