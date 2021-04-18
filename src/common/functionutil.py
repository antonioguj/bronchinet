
from typing import List, Tuple, Dict, Callable, Union, Any
import numpy as np
import itertools
import glob
import pickle
import csv
import os
import re
import shutil
import datetime
import time

from common.exceptionmanager import catch_error_exception


# to interact with the command shell:
def currentdir() -> str:
    return os.getcwd()


def makedir(dirname: str) -> bool:
    dirname = dirname.strip().rstrip("\\")
    if not is_exist_dir(dirname):
        os.makedirs(dirname)
        return True
    else:
        return False


def makelink(src_file: str, dest_link: str) -> None:
    os.symlink(src_file, dest_link)


def get_link_realpath(pathname: str) -> str:
    return os.path.realpath(pathname)


def copydir(src_dir: str, dest_dir: str) -> None:
    shutil.copyfile(src_dir, dest_dir)


def copyfile(src_file: str, dest_file: str) -> None:
    shutil.copyfile(src_file, dest_file)


def movedir(src_dir: str, dest_dir: str) -> None:
    os.rename(src_dir, dest_dir)


def movefile(src_file: str, dest_file: str) -> None:
    os.rename(src_file, dest_file)


def removedir(dirname: str) -> None:
    os.rmdir(dirname)


def removefile(filename: str) -> None:
    os.remove(filename)


def update_dirname(dirname: str) -> str:
    suffix_update_name = 'New%0.2i'
    if is_exist_dir(dirname):
        count = 1
        while True:
            dirname_new = set_dirname_suffix(dirname, suffix_update_name % (count))
            if not is_exist_dir(dirname_new):
                return dirname_new
            count = count + 1   # else: ...keep iterating
    else:
        return dirname


def update_filename(filename: str) -> str:
    suffix_update_name = 'new%0.2i'
    if is_exist_file(filename):
        count = 1
        while True:
            filename_new = set_filename_suffix(filename, suffix_update_name % (count))
            if not is_exist_file(filename_new):
                return filename_new
            count = count + 1   # else: ...keep iterating
    else:
        return filename


def set_dirname_suffix(dirname: str, suffix: str) -> str:
    if dirname.endswith('/'):
        dirname = dirname[:-1]
    return '_'.join([dirname, suffix])


def set_filename_suffix(filename: str, suffix: str) -> str:
    filename_noext, extension = split_filename_extension_recursive(filename)
    return '_'.join([filename_noext, suffix]) + extension


def split_filename_extension(filename: str) -> Tuple[str, str]:
    return os.path.splitext(filename)


def split_filename_extension_recursive(filename: str) -> Tuple[str, str]:
    # accounts for extension that are compound: i.e. '.nii.gz'
    filename_noext, extension = os.path.splitext(filename)
    if extension == '':
        return (filename_noext, extension)
    else:
        sub_filename_noext, sub_extension = split_filename_extension_recursive(filename_noext)
        return (sub_filename_noext, sub_extension + extension)


def is_exist_dir(dirname: str) -> bool:
    return os.path.exists(dirname) and os.path.isdir(dirname)


def is_exist_file(filename: str) -> bool:
    return os.path.exists(filename) and os.path.isfile(filename)


def is_exist_link(filename: str) -> bool:
    return os.path.exists(filename) and os.path.islink(filename)


def is_exist_exec(execname: str) -> bool:
    return os.path.exists(execname) and os.path.isfile(execname) and os.access(execname, os.X_OK)


def is_exists_hexec(execname: str) -> bool:
    return shutil.which(execname) is not None


def join_path_names(pathname_1: str, pathname_2: str) -> str:
    return os.path.join(pathname_1, pathname_2)


def basename(pathname: str) -> str:
    return os.path.basename(pathname)


def basenamedir(pathname: str) -> str:
    if pathname.endswith('/'):
        pathname = pathname[:-1]
    return basename(pathname)


def dirname(pathname: str) -> str:
    return os.path.dirname(pathname)


def dirnamedir(pathname: str) -> str:
    if pathname.endswith('/'):
        pathname = pathname[:-1]
    return dirname(pathname)


def fullpathname(pathname: str) -> str:
    return join_path_names(currentdir(), pathname)


def filename_noext(filename: str, is_split_recursive: bool = True) -> str:
    if is_split_recursive:
        return split_filename_extension_recursive(filename)[0]
    else:
        return split_filename_extension(filename)[0]


def fileextension(filename: str, is_split_recursive: bool = True) -> str:
    if is_split_recursive:
        return split_filename_extension_recursive(filename)[1]
    else:
        return split_filename_extension(filename)[1]


def basename_filenoext(filename: str, is_split_recursive: bool = True) -> str:
    return filename_noext(basename(filename), is_split_recursive)


def list_files_dir(dirname: str, filename_pattern: str = '*', is_check: bool = True) -> List[str]:
    listfiles = sorted(glob.glob(join_path_names(dirname, filename_pattern)))
    if is_check:
        if len(listfiles) == 0:
            message = 'No files found in \'%s\' with \'%s\'' % (dirname, filename_pattern)
            catch_error_exception(message)
    return listfiles


def list_dirs_dir(dirname: str, dirname_pattern: str = '*', is_check: bool = True) -> List[str]:
    return list_files_dir(dirname, dirname_pattern, is_check=is_check)


def list_files_dir_old(dirname: str) -> List[str]:
    listfiles = os.listdir(dirname)
    return [join_path_names(dirname, file) for file in listfiles]


def list_links_dir(dirname: str) -> List[str]:
    listfiles = list_files_dir_old(dirname)
    return [file for file in listfiles if os.path.islink(file)]


# to manipulate substrings in a filename:
def get_substring_filename(filename: str, pattern_search: str) -> Union[str, None]:
    sre_substring_filename = re.search(pattern_search, filename)
    if sre_substring_filename:
        return sre_substring_filename.group(0)
    else:
        return None


def get_regex_pattern_filename(filename: str) -> str:
    # regex pattern by replacing the groups of digits with '[0-9]+'
    basefilename = basename_filenoext(filename)
    list_substrdigits_filename = re.findall('[0-9]+', basefilename)
    if list_substrdigits_filename:
        out_regex_pattern_filename = basefilename
        for i_substrdigits in list_substrdigits_filename:
            out_regex_pattern_filename = out_regex_pattern_filename.replace(i_substrdigits, '[0-9]+', 1)
    else:
        out_regex_pattern_filename = basefilename
    return out_regex_pattern_filename


def find_file_inlist_with_pattern(in_filename: str, inlist_files: List[str], pattern_search: str = None) -> str:
    if not pattern_search:
        # if not input search pattern, get it from the first file in list
        pattern_search = get_regex_pattern_filename(inlist_files[0])

    substring_filename = get_substring_filename(in_filename, pattern_search)
    if not substring_filename:
        message = 'Cannot find a substring with pattern (\'%s\') in the file \'%s\'' % (pattern_search, in_filename)
        catch_error_exception(message)

    is_file_found = False
    for it_file in inlist_files:
        if substring_filename in it_file:
            return it_file

    if not is_file_found:
        dir_files_list = dirname(inlist_files[0])
        list_basefiles = [basename(elem) for elem in inlist_files]
        message = 'Cannot find a file with pattern (\'%s\') as file \'%s\', in the list of files from dir ' \
                  '\'%s\': \'%s\'' % (substring_filename, in_filename, dir_files_list, list_basefiles)
        catch_error_exception(message)


def flatten_listoflists(in_list: List[List[Any]]) -> List[Any]:
    return list(itertools.chain(*in_list))


def find_intersection_2lists(list_1: List[Any], list_2: List[Any]) -> List[Any]:
    return [elem for elem in list_1 if elem in list_2]


def find_intersection_3lists(list_1: List[Any], list_2: List[Any], list_3: List[Any]) -> List[Any]:
    intersection = find_intersection_2lists(list_1, list_2)
    intersection += find_intersection_2lists(list_1, list_3)
    intersection += find_intersection_2lists(list_2, list_3)
    return intersection


# to help parse input arguments:
def str2bool(in_str: str) -> bool:
    return in_str.lower() in ('yes', 'true', 't', '1')


def str2int(in_str: str) -> int:
    return int(in_str)


def str2float(in_str: str) -> float:
    return float(in_str)


def str2list_str(in_str: str) -> List[str]:
    if in_str == '[]':
        return []
    in_str = in_str.replace('[', '').replace(']', '').replace('\'', '').replace(', ', ',')
    list_elems_instr = in_str.split(',')
    return list_elems_instr


def str2list_int(in_str: str) -> List[int]:
    in_str = in_str.replace('[', '').replace(']', '')
    list_elems_instr = in_str.split(',')
    return [str2int(elem) for elem in list_elems_instr]


def str2list_float(in_str: str) -> List[float]:
    in_str = in_str.replace('[', '').replace(']', '')
    list_elems_instr = in_str.split(',')
    return [str2float(elem) for elem in list_elems_instr]


def str2tuple_bool(in_str: str) -> Tuple[int, ...]:
    in_str = in_str.replace('(', '').replace(')', '')
    list_elems_instr = in_str.split(',')
    return tuple([str2bool(elem) for elem in list_elems_instr])


def str2tuple_int(in_str: str) -> Tuple[int, ...]:
    in_str = in_str.replace('(', '').replace(')', '')
    list_elems_instr = in_str.split(',')
    return tuple([str2int(elem) for elem in list_elems_instr])


def str2tuple_float(in_str: str) -> Tuple[float, ...]:
    in_str = in_str.replace('(', '').replace(')', '')
    list_elems_instr = in_str.split(',')
    return tuple([str2float(elem) for elem in list_elems_instr])


def str2tuple_int_none(in_str: str) -> Union[Tuple[int, ...], None]:
    if in_str == 'None':
        return None
    else:
        return str2tuple_int(in_str)


def str2tuple_float_none(in_str: str) -> Union[Tuple[float, ...], None]:
    if in_str == 'None':
        return None
    else:
        return str2tuple_float(in_str)


def list2str(in_list: List[Any]) -> str:
    return '_'.join(str(i) for i in in_list)


def tuple2str(in_tuple: Tuple[Any, ...]) -> str:
    return '_'.join(str(i) for i in list(in_tuple))


def str2list_datatype(in_str: str, elem_type: str) -> List[Any]:
    func_convert_elem = get_func_convert_string_to_datatype(elem_type)
    list_elems_list = split_string_list_or_tuple(in_str)
    return [func_convert_elem(elem) for elem in list_elems_list]


def str2tuple_datatype(in_str: str, elem_type: str) -> Tuple[Any, ...]:
    func_convert_elem = get_func_convert_string_to_datatype(elem_type)
    list_elems_instr = split_string_list_or_tuple(in_str)
    return tuple([func_convert_elem(elem) for elem in list_elems_instr])


def split_string_list_or_tuple(in_str: str) -> List[str]:
    in_str_content = in_str[1:-1].replace(' ', '')
    if ('[' in in_str_content) and (']' in in_str_content):
        in_str_content_split = in_str_content.rsplit('],')
        num_elems = len(in_str_content_split)
        return [(elem + ']') if i < (num_elems - 1) else elem for i, elem in enumerate(in_str_content_split)]
    elif ('(' in in_str_content) and (')' in in_str_content):
        in_str_content_split = in_str_content.rsplit('),')
        num_elems = len(in_str_content_split)
        return [(elem + ')') if i < (num_elems - 1) else elem for i, elem in enumerate(in_str_content_split)]
    else:
        return in_str_content.rsplit(',')


def is_string_bool(in_str: str) -> bool:
    return in_str.lower() in ('yes', 'true', 'no', 'false')


def is_string_int(in_str: str) -> bool:
    return in_str.isdigit()


def is_string_float(in_str: str) -> bool:
    return (in_str.count('.') == 1) and (in_str.replace('.', '', 1).isdigit())


def is_string_list(in_str: str) -> bool:
    return (in_str[0] == '[') and (in_str[-1] == ']')


def is_string_tuple(in_str: str) -> bool:
    return (in_str[0] == '(') and (in_str[-1] == ')')


def get_string_datatype(in_str: str) -> str:
    if is_string_bool(in_str):
        out_datatype = 'bool'
    elif is_string_int(in_str):
        out_datatype = 'int'
    elif is_string_float(in_str):
        out_datatype = 'float'
    elif is_string_list(in_str):
        out_datatype = 'list'
        list_elems_list = split_string_list_or_tuple(in_str)
        out_elem_datatype = get_string_datatype(list_elems_list[0])
        out_datatype += '_' + out_elem_datatype
    elif is_string_tuple(in_str):
        out_datatype = 'tuple'
        list_elems_tuple = split_string_list_or_tuple(in_str)
        out_elem_datatype = get_string_datatype(list_elems_tuple[0])
        out_datatype += '_' + out_elem_datatype
    else:
        out_datatype = 'string'
    return out_datatype


def get_func_convert_string_to_datatype(elem_type: str) -> Callable[[str], Any]:
    if elem_type == 'int':
        return str2int
    elif elem_type == 'float':
        return str2float
    elif 'list' in elem_type and (elem_type[0:5] == 'list_'):
        def func_convert_elem(in_str: str) -> List[Any]:
            return str2list_datatype(in_str, elem_type[5:])
        return func_convert_elem
    elif 'tuple' in elem_type and (elem_type[0:6] == 'tuple_'):
        def func_convert_elem(in_str: str) -> Tuple[Any, ...]:
            return str2tuple_datatype(in_str, elem_type[6:])
        return func_convert_elem
    else:
        # elem_type == 'string':
        def func_dummy(in_str: str) -> str:
            return in_str
        return func_dummy


# to input / output different file formats:
def read_dictionary(filename: str) -> Union[Dict[str, Any], None]:
    extension = fileextension(filename, is_split_recursive=False)
    if extension == '.npy':
        return read_dictionary_numpy(filename)
    if extension == '.pkl':
        return read_dictionary_pickle(filename)
    elif extension == '.csv':
        return read_dictionary_csv(filename)
    else:
        return None


def read_dictionary_numpy(filename: str) -> Dict[str, Any]:
    return dict(np.load(filename, allow_pickle=True).item())


def read_dictionary_pickle(filename: str) -> Dict[str, Any]:
    with open(filename, 'r') as fin:
        return pickle.load(fin)


def read_dictionary_csv(filename: str) -> Dict[str, Any]:
    with open(filename, 'r') as fin:
        reader = csv.reader(fin)
        dict_reader = dict(reader)
        example_value = list(dict_reader.values())[0]
        value_datatype = get_string_datatype(example_value)
        func_convert_values = get_func_convert_string_to_datatype(value_datatype)
        out_dict = {}
        for key, val in dict_reader.items():
            out_dict[key] = func_convert_values(val)
    return out_dict


def read_dictionary_configparams(filename: str) -> Dict[str, str]:
    with open(filename, 'r') as fin:
        out_dict = {}
        for line in fin:
            key, value = line.replace('\n', '').split(' = ')
            out_dict[key] = value
    return out_dict


def save_dictionary(filename: str, in_dictionary: Dict[str, Any]) -> None:
    extension = fileextension(filename, is_split_recursive=False)
    if extension == '.npy':
        save_dictionary_numpy(filename, in_dictionary)
    if extension == '.pkl':
        return save_dictionary_pickle(filename, in_dictionary)
    elif extension == '.csv':
        save_dictionary_csv(filename, in_dictionary)
    else:
        return None


def save_dictionary_numpy(filename: str, in_dictionary: Dict[str, Any]) -> None:
    np.save(filename, in_dictionary)


def save_dictionary_pickle(filename: str, in_dictionary: Dict[str, Any]) -> None:
    with open(filename, 'r') as fout:
        pickle.dump(in_dictionary, fout, pickle.HIGHEST_PROTOCOL)


def save_dictionary_csv(filename: str, in_dictionary: Dict[str, Any]) -> None:
    with open(filename, 'w') as fout:
        writer = csv.writer(fout)
        for key, value in in_dictionary.items():
            writer.writerow([key, value])


def save_dictionary_configparams(filename: str, in_dictionary: Dict[str, Any]) -> None:
    with open(filename, 'w') as fout:
        for key, value in in_dictionary.items():
            strline = '%s = %s\n' % (key, value)
            fout.write(strline)


# timers
def getdatetoday() -> Tuple[int, int, int]:
    today = datetime.date.today()
    return (today.month, today.day, today.year)


def gettimenow() -> Tuple[int, int, int]:
    now = datetime.datetime.now()
    return (now.hour, now.minute, now.second)


class WallClockTime(object):
    def __init__(self) -> None:
        self.start_time = time.time()

    def gettime(self) -> float:
        return time.time() - self.start_time


# others:
def calc_moving_average(in_vals: List[float], size: int) -> List[float]:
    cumsum = np.cumsum(np.insert(in_vals, 0, 0))
    return (cumsum[size:] - cumsum[:-size]) / float(size)
    # return np.convolve(in_vals, np.ones((size,))/size, mode='valid')


class ImagesUtil:
    # size_image: physical dims (dz, dx, dy)
    # shape_image: full shape or image array

    @staticmethod
    def is_without_channels(in_size_image: Union[Tuple[int, int, int], Tuple[int, int]],
                            in_shape_image: Tuple[int, ...]
                            ) -> bool:
        return len(in_shape_image) == len(in_size_image)

    @classmethod
    def get_num_channels(cls, in_size_image: Union[Tuple[int, int, int], Tuple[int, int]],
                         in_shape_image: Tuple[int, ...]
                         ) -> Union[int, None]:
        if cls.is_without_channels(in_size_image, in_shape_image):
            return None
        else:
            return in_shape_image[-1]

    @staticmethod
    def get_shape_channels_first(in_shape_image: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(np.roll(in_shape_image, 1))

    @staticmethod
    def get_shape_channels_last(in_shape_image: Tuple[int, ...]) -> Tuple[int, ...]:
        ndim = len(in_shape_image)
        return tuple(np.roll(in_shape_image, ndim - 1))

    @staticmethod
    def reshape_channels_first(in_image: np.ndarray, is_input_sample: bool = False) -> np.ndarray:
        ndim = len(in_image.shape)
        start_val = 0 if is_input_sample else 1
        return np.rollaxis(in_image, ndim - 1, start_val)

    @staticmethod
    def reshape_channels_last(in_image: np.ndarray, is_input_sample: bool = False) -> np.ndarray:
        ndim = len(in_image.shape)
        axis_val = 0 if is_input_sample else 1
        return np.rollaxis(in_image, axis_val, ndim)


class NetworksUtil:
    # size_input: dims (dz, dx, dy) of input image to network
    # size_output: dims of output image of network, or feature maps
    _num_levels_default = 5
    _num_convols_level_default = 2
    _num_levels_valid_convols_default = 3

    @classmethod
    def calc_size_output_layer_valid_convols(cls, size_input: Union[Tuple[int, int, int], Tuple[int, int]],
                                             num_levels: int = _num_levels_default,
                                             num_convols_level: int = _num_convols_level_default,
                                             num_levels_valid_convols: int = _num_levels_valid_convols_default
                                             ) -> Union[Tuple[int, int, int], Tuple[int, int]]:
        size_output = size_input
        # downsampling levels
        for ilev in range(num_levels - 1):
            for icon in range(num_convols_level):
                if (ilev + 1) <= num_levels_valid_convols:
                    size_output = tuple([cls.calc_size_output_1d_valid_convol(elem) for elem in size_output])
                else:
                    size_output = size_output
            size_output = tuple([cls.calc_size_output_1d_pooling(elem) for elem in size_output])

        # deepest level
        for icon in range(num_convols_level):
            if num_levels <= num_levels_valid_convols:
                size_output = tuple([cls.calc_size_output_1d_valid_convol(elem) for elem in size_output])
            else:
                size_output = size_output

        # upsampling levels
        for ilev in range(num_levels - 2, -1, -1):
            size_output = tuple([cls.calc_size_output_1d_upsample(elem) for elem in size_output])
            for icon in range(num_convols_level):
                if (ilev + 1) <= num_levels_valid_convols:
                    size_output = tuple([cls.calc_size_output_1d_valid_convol(elem) for elem in size_output])
                else:
                    size_output = size_output

        # last layer (1x) convol and does not change dims
        return size_output

    @staticmethod
    def calc_size_output_1d_valid_convol(size_input: int, size_kernel: int = 3) -> int:
        return size_input - size_kernel + 1

    @staticmethod
    def calc_size_output_1d_pooling(size_input: int, size_pool: int = 2) -> int:
        return size_input // size_pool

    @staticmethod
    def calc_size_output_1d_upsample(size_input: int, size_upsample: int = 2) -> int:
        return size_input * size_upsample
