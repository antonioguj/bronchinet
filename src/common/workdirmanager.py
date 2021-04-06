
from common.functionutil import join_path_names, is_exist_dir, is_exist_file, currentdir, makedir, \
    update_dirname, update_filename
from common.exceptionmanager import catch_error_exception


class GeneralDirManager(object):

    def __init__(self, base_path: str) -> None:
        # self._base_path = base_path
        self._base_path = join_path_names(currentdir(), base_path)  # add cwd to get full path
        if not is_exist_dir(self._base_path):
            message = "Base path \'%s\' does not exist" % (self._base_path)
            catch_error_exception(message)

    def get_pathdir_exist(self, rel_path: str) -> str:
        full_path = join_path_names(self._base_path, rel_path)
        if not is_exist_dir(full_path):
            message = "Path \'%s\', does not exist" % (full_path)
            catch_error_exception(message)
        return full_path

    def get_pathdir_new(self, rel_path: str) -> str:
        full_path = join_path_names(self._base_path, rel_path)
        if not is_exist_dir(full_path):
            makedir(full_path)
        return full_path

    def get_pathdir_update(self, rel_path: str) -> str:
        return self.get_pathdir_new(update_dirname(rel_path))

    def get_pathfile_exist(self, filename: str) -> str:
        full_filename = join_path_names(self._base_path, filename)
        if not is_exist_file(full_filename):
            message = "File \'%s\', does not exist" % (full_filename)
            catch_error_exception(message)
        return full_filename

    def get_pathfile_new(self, filename: str) -> str:
        full_filename = join_path_names(self._base_path, filename)
        return full_filename

    def get_pathfile_update(self, filename: str) -> str:
        full_filename_new = join_path_names(self._base_path, update_filename(filename))
        return full_filename_new


class TrainDirManager(GeneralDirManager):
    basedata_rel_path_default = 'BaseData/'

    def __init__(self, base_path: str,
                 basedata_rel_path: str = basedata_rel_path_default
                 ) -> None:
        super(TrainDirManager, self).__init__(base_path)
        self._basedata_relpath = basedata_rel_path
        self._basedata_path = self.get_pathdir_exist(basedata_rel_path)

    def get_datadir_exist(self, rel_path: str) -> str:
        return self.get_pathdir_exist(join_path_names(self._basedata_relpath, rel_path))

    def get_datadir_new(self, rel_path: str) -> str:
        return self.get_pathdir_new(join_path_names(self._basedata_relpath, rel_path))

    def get_datafile_exist(self, filename: str) -> str:
        return self.get_pathfile_exist(join_path_names(self._basedata_relpath, filename))

    def get_datafile_new(self, filename: str) -> str:
        return self.get_pathfile_new(join_path_names(self._basedata_relpath, filename))
