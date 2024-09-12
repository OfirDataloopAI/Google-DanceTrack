from moviepy.editor import ImageSequenceClip
import os
import dtlpy as dl


def download_dataset(dataset: dl.Dataset, download_path: str):
    filters = dl.Filters()
    filters.add(field=dl.FiltersKnownFields.FILENAME, values="*.txt")
    dataset.items.download(local_path=download_path)


def images_to_video(image_folder, output_path, fps=20):
    # Load images from the folder and sort them by name to ensure correct sequence
    image_files = [f"{image_folder}/{img}" for img in sorted(os.listdir(image_folder)) if img.endswith(".jpg")]

    # Create a video clip from the images
    clip = ImageSequenceClip(image_files, fps=fps)

    # Write the output video file in WebM format
    clip.write_videofile(output_path, codec="libvpx-vp9")


def main():
    # Notice: The FPS of the videos is 20

    dl.setenv('rc')
    project = dl.projects.get(project_id="e43f3105-2de2-46a9-b3b8-5f9fcf405adc")
    dataset = project.datasets.get(dataset_id="66e296f67f87ec7d68e59bee")

    download_path = os.path.join(os.getcwd(), f"{dataset.id}_all")

    # Download all items in the dataset
    download_dataset(dataset=dataset, download_path=download_path)

    download_items_path = os.path.join(download_path, "items")

    # Convert images to videos
    all_folders = os.listdir(download_items_path)
    for folder in all_folders:
        print(f"Converting images in folder: {folder}")
        image_folder = os.path.join(download_items_path, folder, "img1")
        output_path = f"{folder}.webm"
        images_to_video(image_folder, output_path)


if __name__ == '__main__':
    main()
