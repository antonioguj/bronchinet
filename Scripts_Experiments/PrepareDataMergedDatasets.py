#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.Constants import *
from Common.FunctionsUtil import *
from Common.WorkDirsManager import *
from collections import OrderedDict
import argparse



def main(args):
    # ---------- SETTINGS ----------
    nameTemplateOutputImagesFiles = 'images_proc-%0.2i.nii.gz'
    nameTemplateOutputLabelsFiles = 'labels_proc-%0.2i.nii.gz'
    nameTemplateOutputExtraLabelsFiles = 'cenlines_proc-%0.2i.nii.gz'
    # ---------- SETTINGS ----------



    list_source_files = ['~/Data/LUVAR_Processed_Fullsize/ImagesWorkData/images_proc-01.nii.gz',
                         '~/Data/LUVAR_Processed_Fullsize/ImagesWorkData/images_proc-02.nii.gz',
                         '~/Data/LUVAR_Processed_Fullsize/ImagesWorkData/images_proc-03.nii.gz',
                         '~/Data/LUVAR_Processed_Fullsize/ImagesWorkData/images_proc-04.nii.gz',
                         '~/Data/LUVAR_Processed_Fullsize/ImagesWorkData/images_proc-05.nii.gz',
                         '~/Data/LUVAR_Processed_Fullsize/ImagesWorkData/images_proc-06.nii.gz',
                         '~/Data/LUVAR_Processed_Fullsize/ImagesWorkData/images_proc-07.nii.gz',
                         '~/Data/LUVAR_Processed_Fullsize/ImagesWorkData/images_proc-08.nii.gz',
                         '~/Data/LUVAR_Processed_Fullsize/ImagesWorkData/images_proc-09.nii.gz',
                         '~/Data/LUVAR_Processed_Fullsize/ImagesWorkData/images_proc-10.nii.gz',
                         '~/Data/LUVAR_Processed_Fullsize/ImagesWorkData/images_proc-11.nii.gz',
                         '~/Data/LUVAR_Processed_Fullsize/ImagesWorkData/images_proc-12.nii.gz',
                         '~/Data/DLCST_Processed_Fullsize/ImagesWorkData/images_proc-01.nii.gz',
                         '~/Data/DLCST_Processed_Fullsize/ImagesWorkData/images_proc-01.nii.gz',
                         '~/Data/DLCST_Processed_Fullsize/ImagesWorkData/images_proc-01.nii.gz',
                         '~/Data/DLCST_Processed_Fullsize/ImagesWorkData/images_proc-01.nii.gz',
                         '~/Data/DLCST_Processed_Fullsize/ImagesWorkData/images_proc-01.nii.gz',
                         '~/Data/DLCST_Processed_Fullsize/ImagesWorkData/images_proc-01.nii.gz',
                         '~/Data/DLCST_Processed_Fullsize/ImagesWorkData/images_proc-01.nii.gz',
                         '~/Data/DLCST_Processed_Fullsize/ImagesWorkData/images_proc-01.nii.gz',
                         '~/Data/DLCST_Processed_Fullsize/ImagesWorkData/images_proc-01.nii.gz',
                         '~/Data/DLCST_Processed_Fullsize/ImagesWorkData/images_proc-01.nii.gz',
                         '~/Data/DLCST_Processed_Fullsize/ImagesWorkData/images_proc-01.nii.gz',
                         '~/Data/DLCST_Processed_Fullsize/ImagesWorkData/images_proc-01.nii.gz',]










if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--datadir', type=str, default=DATADIR)
    # parser.add_argument('--listMergeDataRelDirs', type=list, default=LISTMERGEDATARELDIRS)
    # parser.add_argument('--nameProcImagesRelPath', type=str, default=NAME_PROCIMAGES_RELPATH)
    # parser.add_argument('--nameProcLabelsRelPath', type=str, default=NAME_PROCLABELS_RELPATH)
    # parser.add_argument('--nameProcExtraLabelsRelPath', type=str, default=NAME_PROCEXTRALABELS_RELPATH)
    # parser.add_argument('--nameReferenceKeysFile', type=str, default=NAME_REFERENCEKEYS_FILE)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in sorted(vars(args).iteritems()):
        print("\'%s\' = %s" %(key, value))

    main(args)