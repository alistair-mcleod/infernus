jobname="week2_BG"

savedir="/fred/oz016/alistair/infernus/timeslides_P100_week2_BG"
mkdir -p $savedir

#should be either a path to a file or "None"
#injfile="/fred/oz016/damon/bbh_paper_2/gwtc-3/endo3_bnspop-LIGO-T2100113-v12-1238166018-15843600.hdf5"
injfile="None"

triton_name="triton_$jobname"
cleanup_name="cleanup_$jobname"


sbatch -J $triton_name P100_nosrun.sh $savedir $injfile

sbatch -J $cleanup_name cleanup_job.sh $savedir $injfile
