jobname="week3_inj"

savedir="/fred/oz016/alistair/infernus/timeslides_A100_week3_inj"
mkdir -p $savedir
rm $savedir/* # clean files from previous runs. Be careful you don't delete something you want to keep!

#should be either a path to a file or "None". 
#If "None" it will perform a background run.
#If a path to a file, it will perform an injection run.
injfile="/fred/oz016/damon/bbh_paper_2/gwtc-3/endo3_bnspop-LIGO-T2100113-v12-1238166018-15843600.hdf5"
#injfile="None"

#directory with saved noise.
noisedir="/fred/oz016/alistair/GWSamplegen/noise/O3_third_week_1024"

triton_name="triton_$jobname"
cleanup_name="cleanup_$jobname"

#start the triton server and the workers
sbatch -J $triton_name triton_and_workers_A100.sh $savedir $injfile $noisedir

#start the cleanup job
sbatch -J $cleanup_name cleanup_job.sh $savedir $injfile
