
# --- Core Application Logic ---

class TqdmFileReader:
    """
    A helper class to wrap a file object and update a tqdm progress bar
    based on the number of bytes read.
    """
    def __init__(self, file_obj, pbar):
        self.file_obj = file_obj
        self.pbar = pbar

    def read(self, size=-1):
        chunk = self.file_obj.read(size)
        self.pbar.update(len(chunk))
        return chunk

    def __getattr__(self, name):
        return getattr(self.file_obj, name)