
import glob, os, subprocess




if __name__ == "__main__":

    path_g="Datasets/colmap/colmap/"
    for filename in glob.glob(path_g+'/*'):
        # If this is an image.
        basename = os.path.basename(filename)

        parts = basename.split('_')
        if len(parts)>1 and not '.json' in basename:
            city = parts[0]
            seq_nbr = parts[1]
            path=path_g+city+'_'+seq_nbr
            print(basename)
            status = subprocess.call('colmap model_converter --input_path ' + path + '/sparse/0 --output_path '
                                     + path + ' --output_type TXT',
                                     shell=True)





