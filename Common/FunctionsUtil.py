#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.ErrorMessages import *
import numpy as np
import shutil
import glob
import datetime
import time
import csv
import os
import re


# operations in working directory: mkdir, cp, mv...
def currentdir():
    return os.getcwd()

def removedir(pathname):
    os.rmdir(pathname)

def removefile(filename):
    os.remove(filename)

def makelink(filesrc, linkdest):
    os.symlink(filesrc, linkdest)

def realpathlink(filename):
    return os.path.realpath(filename)

def copydir(dirsrc, dirdest):
    shutil.copyfile(dirsrc, dirdest)

def copyfile(filesrc, filedest):
    shutil.copyfile(filesrc, filedest)

def movedir(pathsrc, pathdest):
    os.rename(pathsrc, pathdest)

def movefile(filesrc, filedest):
    os.rename(filesrc, filedest)

def makedir(pathname):
    pathname = pathname.strip()
    pathname = pathname.rstrip("\\")
    if not isExistdir(pathname):
        os.makedirs(pathname)
        return True
    else:
        return False

def makeUpdatedir(pathname):
    suffix_update = 'NEW%0.2i'
    if isExistdir(pathname):
        count = 1
        while True:
            update_pathname = updatePathnameWithsuffix(pathname, suffix_update % (count))
            if not isExistdir(update_pathname):
                makedir(update_pathname)
                return update_pathname
            # else: ...keep iterating
            count = count + 1
    else:
        makedir(pathname)
        return pathname

def newUpdatefile(filename):
    suffix_update = 'new%0.2i'
    if isExistfile(filename):
        count = 1
        while True:
            update_filename = updateFilenameWithsuffix(filename, suffix_update % (count))
            if not isExistdir(update_filename):
                return update_filename
                # else: ...keep iterating
            count = count + 1
    else:
        return filename

def joinpathnames(pathname, filename):
    return os.path.join(pathname, filename)

def basename(filename):
    return os.path.basename(filename)

def dirnamepathfile(filename):
    return os.path.dirname(filename)

def basenamedir(pathname):
    if pathname.endswith('/'):
        pathname = pathname[:-1]
    return basename(pathname)

def dirnamepathdir(pathname):
    if pathname.endswith('/'):
        pathname = pathname[:-1]
    return dirnamepathfile(pathname)

def fullpathfile(filename):
    return joinpathnames(currentdir(), filename)

def fullpathdir(pathname):
    return joinpathnames(currentdir(), pathname)

def ospathSplitextRecurse(filename):
    #account for extension that are compound: i.e. '.nii.gz'
    filename_noext, extension = os.path.splitext(filename)
    if extension == '':
        return (filename_noext, extension)
    else:
        sub_filename_noext, sub_extension = ospathSplitextRecurse(filename_noext)
        return (sub_filename_noext, sub_extension + extension)

def filenameNoextension(filename, use_recurse_splitext=True):
    if use_recurse_splitext:
        return ospathSplitextRecurse(filename)[0]
    else:
        return os.path.splitext(filename)[0]

def basenameNoextension(filename, use_recurse_splitext=True):
    return filenameNoextension(basename(filename), use_recurse_splitext)

def fileextension(filename, use_recurse_splitext=True):
    if use_recurse_splitext:
        return ospathSplitextRecurse(filename)[1]
    else:
        return os.path.splitext(filename)[1]

def updatePathnameWithsuffix(pathname, suffix):
    if pathname.endswith('/'):
        pathname = pathname[:-1]
    return '_'.join([pathname, suffix])

def updateFilenameWithsuffix(filename, suffix):
    filename_noext, extension = ospathSplitextRecurse(filename)
    new_filename_noext = '_'.join([filename_noext, suffix])
    return new_filename_noext + extension
# ------------------------------------


# find files in working directory
def isExistdir(pathname):
    return os.path.exists(pathname) and os.path.isdir(pathname)

def isExistfile(filename):
    return os.path.exists(filename) and os.path.isfile(filename)

def isExistexec(execname):
    return os.path.exists(execname) and os.path.isfile(execname) and os.access(execname, os.X_OK)

def isExistshexec(execname):
    #return shutil.which(execname) is not None
    return True # reimplement when having python3

def listFilesDir(pathname):
    listfiles = os.listdir(pathname)
    return [joinpathnames(pathname, file) for file in listfiles]

def listLinksDir(pathname):
    listfiles = listFilesDir(pathname)
    return [file for file in listfiles if os.path.islink(file)]

def findFilesDir(filespath, filename_pattern='*'):
    return sorted(glob.glob(joinpathnames(filespath, filename_pattern)))

def findFilesDirAndCheck(filespath, filename_pattern='*'):
    listFiles = findFilesDir(filespath, filename_pattern)
    if (len(listFiles) == 0):
        message = 'no files found in dir \'%s\' with pattern \'%s\'...' %(filespath, filename_pattern)
        CatchErrorException(message)
    return listFiles
# ------------------------------------


# input / output data in disk
def readDictionary(filename):
    extension = fileextension(filename, use_recurse_splitext=False)
    if extension == '.npy':
        return readDictionary_numpy(filename)
    elif extension == '.csv':
        return readDictionary_csv(filename)
    else:
        message = 'unknown extension \'%s\' to read dictionary' %(extension)
        CatchErrorException(message)

def readDictionary_numpy(filename):
    return np.load(filename, allow_pickle=True).item()

def readDictionary_csv(filename):
    with open(filename, 'r') as fin:
        reader = csv.reader(fin)
        raw_out_dict = dict(reader)
        example_value = (raw_out_dict.values())[0]
        value_datatype = getStringDatatype(example_value)
        func_convert_values = getFuncConvertStringDatatype(value_datatype)
        out_dict = {}
        for key, val in raw_out_dict.items():
            out_dict[key] = func_convert_values(val)
        #endfor
        return out_dict

def saveDictionary(filename, dictionary):
    extension = fileextension(filename, use_recurse_splitext=False)
    if extension == '.npy':
        saveDictionary_numpy(filename, dictionary)
    elif extension == '.csv':
        saveDictionary_csv(filename, dictionary)
    else:
        message = 'unknown extension \'%s\' to save dictionary' %(extension)
        CatchErrorException(message)

def saveDictionary_numpy(filename, dictionary):
    np.save(filename, dictionary)

def saveDictionary_csv(filename, dictionary):
    with open(filename, 'w') as fout:
        writer = csv.writer(fout)
        for key, value in dictionary.items():
            writer.writerow([key, value])
        #endfor

def readDictionary_configParams(filename):
    if fileextension(filename, use_recurse_splitext=False) != '.txt':
        message = 'need \'.txt\' to read dictionary'
        CatchErrorException(message)
    else:
        outdictionary = {}
        with open(filename, 'r') as fout:
            for line in fout:
                (key, value) = line.replace('\n','').split(' = ')
                outdictionary[key] = value
            #endfor
        return outdictionary

def saveDictionary_configParams(filename, dictionary):
    if fileextension(filename, use_recurse_splitext=False) != '.txt':
        message = 'need \'.txt\' to save dictionary'
        CatchErrorException(message)
    else:
        with open(filename, 'w') as fout:
            for key, value in dictionary.items():
                strline = '%s = %s\n' %(key, value)
                fout.write(strline)
            #endfor
# ------------------------------------


# manipulate / convert python data types
def isOddIntegerVal(val):
    return val % 2 == 1
def isEvenIntegerVal(val):
    return val % 2 == 0

def isBiggerTuple(var1, var2):
    return all((v_1 > v_2) for (v_1, v_2) in zip(var1, var2))

def isSmallerTuple(var1, var2):
    return all((v_1 < v_2) for (v_1, v_2) in zip(var1, var2))

def isEqualTuple(var1, var2):
    return all((v_1 == v_2) for (v_1, v_2) in zip(var1, var2))

def sumTwoTuples(var1, var2):
    return tuple(a+b for (a,b) in zip(var1, var2))

def substractTwoTuples(var1, var2):
    return tuple(a-b for (a,b) in zip(var1, var2))

def isStrvalBool(strval):
    return strval.lower() in ('yes', 'true', 'no', 'false')

def isStrvalInteger(strval):
    return strval.isdigit()

def isStrvalFloat(strval):
    return (strval.count('.')==1) and (strval.replace('.', '', 1).isdigit())

def isStrvalList(strval):
    return (strval[0] == '[') and (strval[-1] == ']')

def isStrvalTuple(strval):
    return (strval[0] == '(') and (strval[-1] == ')')

def getStringDatatype(strval):
    out_datatype = 0
    if isStrvalBool(strval):
        out_datatype = 'bool'
    elif isStrvalInteger(strval):
        out_datatype = 'int'
    elif isStrvalFloat(strval):
        out_datatype = 'float'
    elif isStrvalList(strval):
        out_datatype = 'list'
        elems_list_split = splitStringListOrTuple(strval)
        out_elem_datatype = getStringDatatype(elems_list_split[0])
        out_datatype += '_'+ out_elem_datatype
    elif isStrvalTuple(strval):
        out_datatype = 'tuple'
        elems_tuple_split = splitStringListOrTuple(strval)
        out_elem_datatype = getStringDatatype(elems_tuple_split[0])
        out_datatype += '_'+ out_elem_datatype
    else:
        message = 'not found data type from string value: \'%s\'' %(strval)
        CatchErrorException(message)
    return out_datatype

def getFuncConvertStringDatatype(elem_type):
    if elem_type == 'int':
        return str2int
    elif elem_type == 'float':
        return str2float
    elif 'list' in elem_type and (elem_type[0:5]=='list_'):
        def func_conv_elem(strval):
            return str2list(strval, elem_type[5:])
        return func_conv_elem
    elif 'tuple' in elem_type and (elem_type[0:6]=='tuple_'):
        def func_conv_elem(strval):
            return str2tuple(strval, elem_type[6:])
        return func_conv_elem
    else:
        message = 'not found data type from string value: \'%s\'' %(elem_type)
        CatchErrorException(message)

def splitStringListOrTuple(strval):
    strval_inside = strval[1:-1].replace(' ','')
    if ('[' in strval_inside) and (']' in strval_inside):
        strval_inside_split = strval_inside.rsplit('],')
        num_elems = len(strval_inside_split)
        return [(elem+']') if i<(num_elems-1) else elem for i, elem in enumerate(strval_inside_split)]
    elif ('(' in strval_inside) and (')' in strval_inside):
        strval_inside_split = strval_inside.rsplit('),')
        num_elems = len(strval_inside_split)
        return [(elem+')') if i<(num_elems-1) else elem for i, elem in enumerate(strval_inside_split)]
    else:
        return strval_inside.rsplit(',')

def str2bool(strval):
    return strval.lower() in ('yes', 'true', 't', '1')

def str2int(strval):
    return int(strval)

def str2float(strval):
    return float(strval)

def str2list(strval, elem_type):
    func_conv_elem = getFuncConvertStringDatatype(elem_type)
    return list([func_conv_elem(elem) for elem in splitStringListOrTuple(strval)])

def str2tuple(strval, elem_type):
    func_conv_elem = getFuncConvertStringDatatype(elem_type)
    return tuple([func_conv_elem(elem) for elem in splitStringListOrTuple(strval)])

def str2tupleint(strval):
    return str2tuple(strval, elem_type='int')

def str2tuplefloat(strval):
    return str2tuple(strval, elem_type='float')

def str2tupleintOrNone(strval):
    return str2tupleint(strval) if (strval != 'None') else None

def str2tuplefloatOrNone(strval):
    return str2tuplefloat(strval) if (strval != 'None') else None

def list2str(list):
    return "_".join(str(i) for i in list)

def tuple2str(tuple):
    return "_".join(str(i) for i in list(tuple))

def splitListInChunks(list, sizechunck):
    listoflists = []
    for i in range(0, len(list), sizechunck):
        listoflists.append( list[i:i+sizechunck] )
    return listoflists

def flattenOutListOfLists(list):
    return reduce(lambda el1, el2: el1 + el2, list)

def mergeTwoListsIntoDictoinary(list1, list2):
    new_dict = {}
    map(lambda key, val: new_dict.update({key: val}), list1, list2)
    return new_dict

def findElementsSubstringInListStrings(list, pattern):
    return [elem for elem in list if pattern in elem]

def findIntersectionTwoLists(list1, list2):
    return [elem for elem in list1 if elem in list2]

def findIntersectionThreeLists(list1, list2, list3):
    intersection  = findIntersectionTwoLists(list1, list2)
    intersection += findIntersectionTwoLists(list1, list3)
    intersection += findIntersectionTwoLists(list2, list3)
    return intersection

# def findCommonElementsTwoLists_Option1(list1, list2):
#     if len(list1) == 0 or len(list2) == 0:
#         return False
#     elif type(list1[0]) != type(list2[0]):
#         return False
#     else:
#         list_common_elems = []
#         for elem1 in list1:
#             if elem1 in list2:
#                 list_common_elems.append(elem1)
#         #endfor
#         return list_common_elems
#
# def findCommonElementsTwoLists_Option2(list1, list2):
#     if len(list1) == 0 or len(list2) == 0:
#         return False
#     elif type(list1[0]) != type(list2[0]):
#         return False
#     else:
#         setlist1 = set([tuple(elem) for elem in list1])
#         setlist2 = set([tuple(elem) for elem in list2])
#         return list([elem for elem in setlist1 & setlist2])
# ------------------------------------


# timers
def getdatetoday():
    today = datetime.date.today()
    return (today.month, today.day, today.year)

def gettimenow():
    now = datetime.datetime.now()
    return (now.hour, now.minute, now.second)

class WallClockTime(object):
    def __init__(self):
        self.start_time = time.time()
    def compute(self):
        return time.time() - self.start_time
# ------------------------------------


# others util
def parseListarg(args):
    return args.replace('[','').replace(']','').split(',')

def getFormatFileExtension(formatfile):
    if formatfile=='dicom':
        return '.dcm'
    elif formatfile=='nifti':
        return '.nii'
    elif formatfile=='nifti_gz':
        return '.nii.gz'
    elif formatfile=='numpy':
        return '.npy'
    elif formatfile=='numpy_gzbi':
        return '.npz'
    elif formatfile=='numpy_gz':
        return '.npy.gz'
    elif formatfile=='hdf5':
        return '.hdf5'
    else:
        return False

def findFileWithSamePrefix(prefix_file, list_infiles):
    for iter_file in list_infiles:
        if prefix_file in iter_file:
            return iter_file
    #endfor
    message = 'not found file with same prefix \'%s\' in list files: \'%s\'' %(prefix_file, list_infiles)
    CatchErrorException(message)

def getSubstringPatternFilename(filename, substr_pattern):
    return re.search(substr_pattern, filename).group(0)

def findFileWithSamePrefixPattern(source_file, list_infiles, prefix_pattern=None):
    if not prefix_pattern:
        prefix_pattern = getFilePrefixPattern(list_infiles[0])

    prefix_casename = getSubstringPatternFilename(source_file, prefix_pattern)
    for iter_file in list_infiles:
        if prefix_casename in iter_file:
            return iter_file
    #endfor
    message = 'not found file with same prefix pattern in \'%s\' in list files: \'%s\'' %(source_file, list_infiles)
    CatchErrorException(message)

def findListFilesWithSamePrefixPattern(source_file, list_infiles, prefix_pattern=None):
    if not prefix_pattern:
        prefix_pattern = getFilePrefixPattern(list_infiles[0])

    prefix_casename = getSubstringPatternFilename(source_file, prefix_pattern)
    list_out_files = []
    for iter_file in list_infiles:
        if prefix_casename in iter_file:
            list_out_files.append(iter_file)
    #endfor
    if len(list_out_files)==0:
        message = 'not found files with same prefix pattern in \'%s\' in list files: \'%s\'' % (source_file, list_infiles)
        CatchErrorException(message)
    else:
        return list_out_files

def getIndexOriginImagesFile(images_file, beginString='images', firstIndex='0'):
    pattern = beginString + '-[0-9]*'
    if bool(re.match(pattern, images_file)):
        index_origin = int(re.search(pattern, images_file).group(0)[-2:])
        return index_origin - int(firstIndex)
    else:
        return False

def getFilePrefixPattern(in_file):
    base_infile = basenameNoextension(in_file)
    infile_prefix = base_infile.split('_')[0]
    infile_prefix_pattern = ''.join(['[0-9]' if s.isdigit() else s for s in infile_prefix])
    if infile_prefix != base_infile:
        infile_prefix_pattern += '_'
    return infile_prefix_pattern

def getIntegerInString(in_name):
    return int(re.findall('\d+', in_name)[0])
# ------------------------------------
