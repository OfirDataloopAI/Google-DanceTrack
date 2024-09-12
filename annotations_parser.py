import dtlpy as dl
import os
import pathlib
import json
from tqdm import tqdm


def annotations_uploader(dataset: dl.Dataset):
    download_path = os.path.join(os.getcwd(), dataset.id)
    os.makedirs(download_path, exist_ok=True)

    filters = dl.Filters()
    filters.add(field=dl.FiltersKnownFields.FILENAME, values="*.txt")
    dataset.items.download(local_path=download_path, filters=filters)
    dataset.download_annotations(local_path=download_path, filters=filters)

    text_files = pathlib.Path(download_path).rglob('*.txt')
    for text_file in text_files:
        # Read annotations from text file
        print(f"Processing file: {text_file}")
        with open(text_file, 'r') as f:
            annotations = f.readlines()
        annotations = [annotation.strip() for annotation in annotations]
        # print(f"Annotations: {annotations}")

        # Read images dir from annotations json file
        text_file_json_filepath = str(text_file).replace('items', 'json')
        text_file_json_filepath = text_file_json_filepath.replace('.txt', '.json')
        with open(text_file_json_filepath, 'r') as f:
            text_file_json = json.load(f)
        images_dir = text_file_json["dir"].replace("gt", "img1")

        # Get all video frames
        filters = dl.Filters()
        filters.add(field=dl.FiltersKnownFields.DIR, values=images_dir)
        filters.sort_by(field=dl.FiltersKnownFields.FILENAME, value=dl.FiltersOrderByDirection.ASCENDING)
        items = list(dataset.items.list(filters=filters).all())

        # Create annotation builders
        item_builders: [dl.AnnotationCollection] = list()
        item: dl.Item
        for item in items:
            builder = item.annotations.builder()
            item_builders.append(builder)

        # Add annotations to builders
        for annotation in annotations:
            # Annotation format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, 1, 1, 1
            frame, object_id, bb_left, bb_top, bb_width, bb_height, _, _, _ = annotation.split(',')
            bb_bottom = int(bb_top) + int(bb_height)
            bb_right = int(bb_left) + int(bb_width)

            item_builders[int(frame) - 1].add(
                annotation_definition=dl.Box(
                    label="person",
                    top=bb_top,
                    left=bb_left,
                    bottom=bb_bottom,
                    right=bb_right
                ),
                object_id=object_id
            )

        # Upload annotations
        print(f"Uploading annotations for {len(item_builders)} items...")
        for builder in tqdm(item_builders):
            builder.upload()

            exit()


def main():
    dl.setenv('rc')
    project = dl.projects.get(project_id="e43f3105-2de2-46a9-b3b8-5f9fcf405adc")
    dataset = project.datasets.get(dataset_id="66e2a3807f87ec4cf2e5bf43")

    annotations_uploader(dataset=dataset)


if __name__ == '__main__':
    main()
