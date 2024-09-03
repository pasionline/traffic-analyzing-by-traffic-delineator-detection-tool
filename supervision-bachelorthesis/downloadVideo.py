import os
import yt_dlp as youtube_dl


def download_video(url, download_path='.', quality='best'):
    """
    Downloads a YouTube video using yt-dlp.

    :param url: The URL of the YouTube video.
    :param download_path: The directory where the video will be saved.
    :param quality: The format in which to download the video (e.g., 'best', 'worst', 'mp4', etc.).
    """
    ydl_opts = {
        'quality': quality,
        'outtmpl': os.path.join(download_path, '%(title)s.%(ext)s'),
        'noplaylist': True,
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


if __name__ == "__main__":
    # Example usage
    video_url = input("Enter the YouTube video URL: ")
    download_directory = input("Enter the download directory (default is current directory): ")

    if not download_directory:
        download_directory = '.'

    video_format = input("Enter the video format (default is 'best'): ")

    if not video_format:
        video_format = 'best'

    download_video(video_url, download_directory, video_format)
    print("Download complete!")
