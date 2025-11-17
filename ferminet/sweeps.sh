
#!/bin/bash

sbatch --job-name=opt_kfac --output=doc_kfac.out opt_sweep.sh 'kfac'
sbatch --job-name=opt_adam --output=doc_adam.out opt_sweep.sh 'adam'
sbatch --job-name=opt_minsr --output=doc_minsr.out opt_sweep.sh 'minsr'