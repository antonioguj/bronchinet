
from typing import List, Tuple, Dict, Callable, Any
import numpy as np
import itertools
import shutil
import glob
import datetime
import time
import csv
import os
import re

from common.exceptionmanager import catch_error_exception, catch_warning_exception


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

def basename_dir(pathname: str) -> str:
    if pathname.endswith('/'):
        pathname = pathname[:-1]
    return basename(pathname)

def dirname(pathname: str) -> str:
    return os.path.dirname(pathname)

def dirname_dir(pathname: str) -> str:
    if pathname.endswith('/'):
        pathname = pathname[:-1]
    return dirname(pathname)

def fullpathname(pathname: str) -> str:
    return join_path_names(currentdir(), pathname)

def filename_noext(filename: str, is_split_recursive: bool=True) -> str:
    if is_split_recursive:
        return split_filename_extension_recursive(filename)[0]
    else:
        return split_filename_extension(filename)[0]

def fileextension(filename: str, is_split_recursive: bool=True) -> str:
    if is_split_recursive:
        return split_filename_extension_recursive(filename)[1]
    else:
        return split_filename_extension(filename)[1]

def basename_file_noext(filename: str, is_split_recursive: bool=True) -> str:
    return filename_noext(basename(filename), is_split_recursive)

def list_files_dir(dirname: str, filename_pattern: str='*', is_check: bool=True) -> List[str]:
    listfiles = sorted(glob.glob(join_path_names(dirname, filename_pattern)))
    if is_check:
        if (len(listfiles) == 0):
            message = 'No files found in \'%s\' with \'%s\'' % (dirname, filename_pattern)
            catch_error_exception(message)
    return listfiles

def list_files_dir_old(dirname: str) -> List[str]:
    listfiles = os.listdir(dirname)
    return [join_path_names(dirname, file) for file in listfiles]

def list_links_dir(dirname: str) -> List[str]:
    listfiles = list_files_dir_old(dirname)
    return [file for file in listfiles if os.path.islink(file)]


# to manipulate strings in filename:
def get_substring_filename(filename: str, substr_pattern: str) -> str:
    return re.search(substr_pattern, filename).group(0)

def get_prefix_pattern_filename(filename: str, char_split_name: str='_') -> str:
    basefilename_noext = basename_file_noext(filename)
    prefix_file = basefilename_noext.split(char_split_name)[0]
    prefix_pattern = ''.join(['[0-9]' if s.isdigit() else s for s in prefix_file])

    if prefix_file != basefilename_noext:
        prefix_pattern += char_split_name
    return prefix_pattern

def find_file_inlist_same_prefix(in_filename: str,
                                 list_files: List[str],
                                 prefix_pattern: str=None) -> str:
    if not prefix_pattern:
        # if not input pattern, get it from the first file in list
        prefix_pattern = get_prefix_pattern_filename(list_files[0])
    prefix_filename = get_substring_filename(in_filename, prefix_pattern)

    for it_file in list_files:
        if prefix_filename in it_file:
            return it_file

    message = 'No file found with prefix of \'%s\', computed as \'%s\', in list of files: \'%s\'' % (in_filename, prefix_filename, list_files)
    catch_error_exception(message)

def find_listfiles_inlist_same_prefix(in_filename: str,
                                      list_files: List[str],
                                      prefix_pattern: str=None) -> List[str]:
    if not prefix_pattern:
        # if not input pattern, get it from the first file in list
        prefix_pattern = get_prefix_pattern_filename(list_files[0])
    prefix_filename = get_substring_filename(in_filename, prefix_pattern)

    list_out_files = []
    for it_file in list_files:
        if prefix_filename in it_file:
            list_out_files.append(it_file)

    if len(list_out_files) == 0:
        message = 'No files found with prefix of \'%s\', computed as \'%s\', in list of files: \'%s\'' % (in_filename, prefix_filename, list_files)
        catch_error_exception(message)
    else:
        return list_out_files

def flatten_listoflists(in_list: List[List[Any]]) -> List[Any]:
    return list(itertools.chain(*in_list))

def find_intersection_2lists(list_1: List[Any],
                             list_2: List[Any]) -> List[Any]:
    return [elem for elem in list_1 if elem in list_2]

def find_intersection_3lists(list_1: List[Any],
                             list_2: List[Any],
                             list_3: List[Any]) -> List[Any]:
    intersection  = find_intersection_2lists(list_1, list_2)
    intersection += find_intersection_2lists(list_1, list_3)
    intersection += find_intersection_2lists(list_2, list_3)
    return intersection


# to help parse input arguments:
def str2bool(in_str: str) -> bool:
    return in_str.lower() in ('yes', 'true', 't', '1')

def is_string_bool(in_str: str) -> bool:
    return in_str.lower() in ('yes', 'true', 'no', 'false')

def str2int(in_str: str) -> int:
    return int(in_str)

def is_string_int(in_str: str) -> bool:
    return in_str.isdigit()

def str2float(in_str: str) -> float:
    return float(in_str)

def is_string_float(in_str: str) -> bool:
    return (in_str.count('.')==1) and (in_str.replace('.', '', 1).isdigit())

def str2list_string(in_str: str) -> List[str]:
    return in_str.replace('[','').replace(']','').split(',')

def str2list_datatype(in_str: str, elem_type: str) -> List[Any]:
    func_convert_elem = get_func_convert_string_to_datatype(elem_type)
    list_elems_list = split_string_list_or_tuple(in_str)
    return list([func_convert_elem(elem) for elem in list_elems_list])

def is_string_list(in_str: str) -> bool:
    return (in_str[0] == '[') and (in_str[-1] == ']')

def str2tuple_datatype(in_str: str, elem_type: str) -> Tuple[Any, ...]:
    func_convert_elem = get_func_convert_string_to_datatype(elem_type)
    list_elems_tuple = split_string_list_or_tuple(in_str)
    return tuple([func_convert_elem(elem) for elem in list_elems_tuple])

def str2tuple_int(in_str: str) -> Tuple[Any, ...]:
    if in_str == 'None':
        return None
    else:
        return str2tuple_datatype(in_str, elem_type='int')

def str2tuple_float(in_str: str) -> Tuple[Any, ...]:
    if in_str == 'None':
        return None
    else:
        return str2tuple_datatype(in_str, elem_type='float')

def is_string_tuple(in_str: str) -> bool:
    return (in_str[0] == '(') and (in_str[-1] == ')')

def list2str(in_list: List[Any]) -> str:
    return '_'.join(str(i) for i in in_list)

def tuple2str(in_tuple: Tuple[Any, ...]) -> str:
    return '_'.join(str(i) for i in list(in_tuple))

def split_string_list_or_tuple(in_str: str) -> List[str]:
    in_str_content = in_str[1:-1].replace(' ','')
    if ('[' in in_str_content) and (']' in in_str_content):
        in_str_content_split = in_str_content.rsplit('],')
        num_elems = len(in_str_content_split)
        return [(elem+']') if i<(num_elems-1) else elem for i, elem in enumerate(in_str_content_split)]
    elif ('(' in in_str_content) and (')' in in_str_content):
        in_str_content_split = in_str_content.rsplit('),')
        num_elems = len(in_str_content_split)
        return [(elem+')') if i<(num_elems-1) else elem for i, elem in enumerate(in_str_content_split)]
    else:
        return in_str_content.rsplit(',')

def get_string_datatype(in_str: str) -> str:
    out_datatype = 0
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
        message = 'not found datatype from string: \'%s\'' % (in_str)
        catch_error_exception(message)
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
        message = 'not found datatype from elem: \'%s\'' % (elem_type)
        catch_error_exception(message)


# to input / output different file formats:
def read_dictionary(filename: str) -> Dict[str, Any]:
    extension = fileextension(filename, is_split_recursive=False)
    if extension == '.npy':
        return read_dictionary_numpy(filename)
    elif extension == '.csv':
        return read_dictionary_csv(filename)
    else:
        message = 'Unknown file extension \'%s\' to read dictionary' % (extension)
        catch_error_exception(message)

def read_dictionary_numpy(filename: str) -> Dict[str, Any]:
    return np.load(filename, allow_pickle=True).item()

def read_dictionary_csv(filename: str) -> Dict[str, Any]:
    with open(filename, 'r') as fin:
        reader = csv.reader(fin)
        raw_out_dict = dict(reader)
        example_value = (raw_out_dict.values())[0]
        value_datatype = get_string_datatype(example_value)
        func_convert_values = get_func_convert_string_to_datatype(value_datatype)
        out_dict = {}
        for key, val in raw_out_dict.items():
            out_dict[key] = func_convert_values(val)

    return out_dict

def read_dictionary_configparams(filename: str) -> Dict[str, str]:
    with open(filename, 'r') as fin:
        out_dict = {}
        for line in fin:
            key, value = line.replace('\n','').split(' = ')
            out_dict[key] = value

    return out_dict

def save_dictionary(filename: str, in_dict: Dict[str, Any]) -> None:
    extension = fileextension(filename, is_split_recursive=False)
    if extension == '.npy':
        save_dictionary_numpy(filename, in_dict)
    elif extension == '.csv':
        save_dictionary_csv(filename, in_dict)
    else:
        message = 'Unknown file extension \'%s\' to save dictionary' % (extension)
        catch_error_exception(message)

def save_dictionary_numpy(filename: str, in_dict: Dict[str, Any]) -> None:
    np.save(filename, in_dict)

def save_dictionary_csv(filename: str, in_dict: Dict[str, Any]) -> None:
    with open(filename, 'w') as fout:
        writer = csv.writer(fout)
        for key, value in in_dict.items():
            writer.writerow([key, value])

def save_dictionary_configparams(filename: str, in_dict: Dict[str, Any]) -> None:
    with open(filename, 'w') as fout:
        for key, value in in_dict.items():
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
class ImagesUtil:
    # size_image: physical dims (dz, dx, dy)
    # shape_image: full shape or image array

    @staticmethod
    def is_image_without_channels(in_size_image: Tuple[int, ...], in_shape_image: Tuple[int, ...]) -> bool:
        return len(in_shape_image) == len(in_size_image)

    @classmethod
    def get_num_channels_image(cls, in_size_image: Tuple[int, ...], in_shape_image: Tuple[int, ...]) -> int:
        if cls.is_image_without_channels(in_size_image, in_shape_image):
            return None
        else:
            return in_shape_image[-1]