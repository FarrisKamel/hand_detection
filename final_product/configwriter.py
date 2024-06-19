from configparser import ConfigParser

config = ConfigParser()

config["DEFAULT"] = {
    "image_blurriness_threshold": 50
}

with open("parser.ini", "w") as f:
    config.write(f)
