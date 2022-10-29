import config
from  models import *
import os
con = config.Config()
con.set_in_path("./benchmarks/WN18RR/")
con.set_dataset_name("WN18RR")
con.set_model_name("BoxE")
con.set_start_from("./checkpoint/BoxE-499.ckpt")
con.set_work_threads(8)
con.set_train_times(4) # 4000
con.set_nbatches(20)  #10
con.set_alpha(.35) # 0.035
con.set_bern(1)
con.set_dimension(100)#200
con.set_lmbda(0.001)
con.set_margin(1.0)
con.set_ent_neg_rate(10)
con.set_rel_neg_rate(0)
con.set_opt_method("adam") #adagrad
con.set_save_steps(4) # 4000
con.set_valid_steps(4) # 4000
con.set_early_stopping_patience(10)
con.set_checkpoint_dir("./checkpoint")
con.set_result_dir("./result")
con.set_test_link(True)
con.set_test_triple(True)
con.init()
con.set_train_model(BoxE)
con.train()
os.environ["CUDA_VISIBLE_DEVICES"]="3"
