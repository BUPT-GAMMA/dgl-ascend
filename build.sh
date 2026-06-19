rm -rf ~/dgl-ascend/build
bash script/build_dgl_ascend.sh
cd python
pip install -e .
msprof --output=$HOME/msprof_segment_reduce \
  --application="python $HOME/dgl-ascend/tests/ascend/test_segment_reduce_npu.py"