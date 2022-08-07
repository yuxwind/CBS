set -x
for i in `seq 0 81`; do
    sh tests/mnist_tests.sh $i
done
set +x
