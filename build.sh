rm -rf build
bash script/build_dgl_ascend.sh
cd python
pip install -e .