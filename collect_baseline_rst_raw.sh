DIR=$1
parent_dir=`dirname "$1"`
# clear the previous results
if [ "$#" -eq 2 ]; then
    grep 'top5 are'  $DIR/*_0seed/sparsity*.log >  $parent_dir/all.top1_top5
else
    grep 'top5 are'  $DIR/*_0seed/sparsity*.log >>  $parent_dir/all.top1_top5
fi
grep 'top1 are'  $DIR/*_0seed/sparsity*.log >> $parent_dir/all.top1_top5
grep 'top5 are'  $DIR/*_1seed/sparsity*.log >> $parent_dir/all.top1_top5 
grep 'top1 are'  $DIR/*_1seed/sparsity*.log >> $parent_dir/all.top1_top5
grep 'top1 are'  $DIR/*_2seed/sparsity*.log >> $parent_dir/all.top1_top5
grep 'top5 are'  $DIR/*_2seed/sparsity*.log >> $parent_dir/all.top1_top5
grep 'top5 are'  $DIR/*_3seed/sparsity*.log >> $parent_dir/all.top1_top5
grep 'top1 are'  $DIR/*_3seed/sparsity*.log >> $parent_dir/all.top1_top5
grep 'top1 are'  $DIR/*_4seed/sparsity*.log >> $parent_dir/all.top1_top5
grep 'top5 are'  $DIR/*_4seed/sparsity*.log >> $parent_dir/all.top1_top5
#mv $1/all.top1_top5 $1.all.top1_top5
