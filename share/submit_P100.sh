jobname="P100_BG"

jsonfile="bg_P100.json"

savedir=$(cat $jsonfile | grep -F '"save_dir": ' | sed -e 's/"save_dir": //' | tr -d '",')

mkdir -p $savedir
rm -f $savedir/* # clean files from previous runs. Be careful you don't delete something you want to keep!

#should be either a path to a file or "None". 
#If "None" it will perform a background run.
#If a path to a file, it will perform an injection run.
#injfile="/fred/oz016/damon/bbh_paper_2/gwtc-3/endo3_bnspop-LIGO-T2100113-v12-1238166018-15843600.hdf5"

injfile=$(cat $jsonfile | grep -F '"injfile": ' | sed -e 's/"injfile": //' | tr -d '",')


triton_name="triton_$jobname"
cleanup_name="cleanup_$jobname"

#start the triton server and the workers
sbatch -J $triton_name triton_and_workers_P100.sh $savedir $jsonfile

#start the cleanup job
sbatch -J $cleanup_name cleanup_job.sh $savedir $injfile
