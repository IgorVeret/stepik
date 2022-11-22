from pytube import YouTube
url = 'https://www.youtube.com/watch?v=xgVr9vISm8w&ab_channel=DeepLearningSchool'
my_video = YouTube(url)
print(my_video.title)
print(my_video.thumbnail_url)
my_video = my_video.streams.get_highest_resolution()
# for stream in my_video.streams:
#     print(stream)
my_video.download()