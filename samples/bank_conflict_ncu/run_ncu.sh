ncu -k regex:"load" \
--metrics \
smsp__inst_executed_op_shared_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
smsp__inst_executed_op_shared_st.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
-f python test_for_ncu.py
