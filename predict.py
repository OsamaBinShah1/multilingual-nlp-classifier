import argparse, json, os
from model.classifier import TextClassifier

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--model_path", default="./saved_model")
    args = p.parse_args()

    label_map_path = os.path.join(args.model_path, "label_map.json")
    clf = TextClassifier.load(args.model_path)
    if os.path.exists(label_map_path):
        with open(label_map_path) as f:
            lm = json.load(f)
        clf.id2label = {int(k): v for k, v in lm["id2label"].items()}

    results = clf.predict(args.text)
    for r in results:
        print(f"Text:       {r['text']}")
        print(f"Label:      {r['label']}")
        print(f"Confidence: {r['confidence']:.1%}")

if __name__ == "__main__":
    main()
