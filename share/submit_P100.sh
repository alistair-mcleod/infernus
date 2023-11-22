jobname="week3_P"

savedir="/fred/oz016/alistair/infernus/timeslides_P100_week3_inj"
mkdir -p $savedir
rm $savedir/* # clean files from previous runs. Be careful you don't delete something you want to keep!

#should be either a path to a file or "None"
injfile="/fred/oz016/damon/bbh_paper_2/gwtc-3/endo3_bnspop-LIGO-T2100113-v12-1238166018-15843600.hdf5"
#injfile="None"

noisedir="/fred/oz016/alistair/GWSamplegen/noise/O3_third_week_1024"

triton_name="triton_$jobname"
cleanup_name="cleanup_$jobname"


sbatch -J $triton_name P100_nosrun.sh $savedir $injfile $noisedir

sbatch -J $cleanup_name cleanup_job.sh $savedir $injfile
