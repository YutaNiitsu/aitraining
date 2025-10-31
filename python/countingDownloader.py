from icrawler.downloader import ImageDownloader

# ダウンロード数をカウントする
class CountingDownloader(ImageDownloader):
    def __init__(self, storage, thread_num, signal, session, *args, **kwargs):
        super().__init__(storage=storage, thread_num=thread_num, signal=signal, session=session, *args, **kwargs)
        self.saved_count = 0

    def download(self, task, default_ext, timeout=5, **kwargs):
        try:
            result = super().download(task, default_ext, timeout, **kwargs)
            if result:
                self.saved_count += 1
            return result
        except Exception as e:
            print(f"Download error: {e}")
            return False