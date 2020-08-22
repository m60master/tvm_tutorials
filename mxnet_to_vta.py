# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
from mxnet.gluon.model_zoo import vision
import numpy as np
from os.path import join, isfile
from matplotlib import pyplot as plt
from PIL import Image

import pdb
import topi
import tvm
from tvm import te
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_runtime, util, download

from tvm.autotvm.measure.measure_methods import request_remote
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

import vta
from vta.testing import simulator
from vta.top import graph_pack

#################################################################
# Compile network
# ---------------
# Perform vta-specific compilation with Relay from a Gluon model

def compile_network(env, target, model, start_pack, stop_pack, start_name_idx, stop_name_idx):

    # Populate the shape and data type dictionary
    dtype_dict = {"data": 'float32'}
    shape_dict = {"data": (env.BATCH, 3, 224, 224)}

    # Get off the shelf gluon model, and convert to relay
    gluon_model = vision.get_model(model, pretrained=True)
    mod, params = relay.frontend.from_mxnet(gluon_model, shape_dict)

    # Update shape and type dictionary
    shape_dict.update({k: v.shape for k, v in params.items()})
    dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

    # Perform quantization in Relay
    # Note: We set opt_level to 3 in order to fold batch norm
    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
            mod = relay.quantize.quantize(mod, params=params)
            
    # Perform graph packing and constant folding for VTA target
    if target.device_name == "vta":
        assert env.BLOCK_IN == env.BLOCK_OUT
        relay_prog = graph_pack(mod["main"],
                                env.BATCH,
                                env.BLOCK_OUT,
                                env.WGT_WIDTH,
                                start_name=start_pack,
                                stop_name=stop_pack,
                                start_name_idx=start_name_idx,
                                stop_name_idx=stop_name_idx
                                    )
    return relay_prog, params


###########################################
# Set Tuning Options
# ------------------
# Before tuning, we should apply some configurations.
# Here we use an Pynq-Z1 board as an example.

# Tracker host and port can be set by your environment
tracker_host = os.environ.get("TVM_TRACKER_HOST", '0.0.0.0')
tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))

# Load VTA parameters from the 3rdparty/vta-hw/config/vta_config.json file
env = vta.get_env()

# This target is used for cross compilation. You can query it by :code:`gcc -v` on your device.
# Set ``device=arm_cpu`` to run inference on the CPU
# or ``device=vta`` to run inference on the FPGA.
device = "vta"
target = env.target if device == "vta" else env.target_vta_cpu

# Name of Gluon model to compile
# The ``start_pack`` and ``stop_pack`` labels indicate where
# to start and end the graph packing relay pass: in other words
# where to start and finish offloading to VTA.

#network = "vgg16"
#start_pack = "nn.conv2d"
#stop_pack = "nn.max_pool2d"
#start_name_idx = 0
#stop_name_idx = 111

#network = "squeezenet1.0"
#start_pack = "nn.conv2d"
#stop_pack = "nn.avg_pool2d"
#start_name_idx = 0
#stop_name_idx = 255

#network = "alexnet"
#start_pack = "nn.conv2d"
#stop_pack = "nn.max_pool2d"
#start_name_idx = 0
#stop_name_idx = 42

#network = "resnet50_v1"
#start_pack = "nn.max_pool2d"
#stop_pack = "nn.global_avg_pool2d"

network = "resnet18_v2"
start_pack = "nn.conv2d"
stop_pack = "nn.global_avg_pool2d"
start_name_idx = 1
stop_name_idx = 231


# Tuning option
log_file = "%s.%s.log" % (device, network)
tuning_option = {
    'log_filename': log_file,

    'tuner': 'xgb',
    'n_trial': 1,
    'early_stopping': None,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.RPCRunner(env.TARGET,
                                 host=tracker_host,
                                 port=tracker_port,
                                 number=5,
                                 timeout=60,
                                 check_correctness=True),
    ),
}

###################################################################
# Begin Tuning
# ------------
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=False):

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'xgb_knob':
            tuner_obj = XGBTuner(tsk, loss_type='rank', feature_type='knob')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=tsk_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)
                       ])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)

########################################################################
# Register VTA-specific tuning tasks


def register_vta_tuning_tasks():
    from tvm.autotvm.task import TaskExtractEnv

    @tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
    def my_clip(x, a_min, a_max):
        """Unlike topi's current clip, put min and max into two stages."""
        const_min = tvm.tir.const(a_min, x.dtype)
        const_max = tvm.tir.const(a_max, x.dtype)
        x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
        x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
        return x

    # init autotvm env to register VTA operator
    TaskExtractEnv()

    @autotvm.template("conv2d_packed.vta")
    def _topi_nn_conv2d(*args, **kwargs):
        assert not kwargs, "Do not support kwargs in template function call"
        A, W = args[:2]

        with tvm.target.vta():
            res = vta.top.conv2d_packed(*args, **kwargs)
            res = topi.right_shift(res, 8)
            res = my_clip(res, 0, 127)
            res = topi.cast(res, "int8")

        if tvm.target.Target.current().device_name == 'vta':
            s = vta.top.schedule_conv2d_packed([res])
        else:
            s = te.create_schedule([res.op])
        return s, [A, W, res]


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.

def tune_and_evaluate(tuning_opt):

    if env.TARGET != "sim":
        # Get remote from fleet node
        remote = autotvm.measure.request_remote(env.TARGET,
                                                tracker_host,
                                                tracker_port,
                                                timeout=10000)
        # Reconfigure the JIT runtime and FPGA.
        vta.reconfig_runtime(remote)
        vta.program_fpga(remote, bitstream=None)
    else:
        # In simulation mode, host the RPC server locally.
        remote = rpc.LocalSession()

    # Register VTA tuning tasks
    register_vta_tuning_tasks()

    # Perform task extraction on Relay program
    print("Extract tasks...")
    relay_prog, params = compile_network(env, target, network, start_pack, stop_pack,
                                                                start_name_idx, stop_name_idx)
    
    mod = tvm.IRModule.from_expr(relay_prog)
    tasks = autotvm.task.extract_from_program(mod,
                                              params=params,
                                              ops=(relay.op.get("nn.conv2d"),),
                                              target=target,
                                              target_host=env.target_host)


    # filter out non-packed conv2d task
    tasks = list(filter(lambda t: len(t.args[0][1]) > 4, tasks))

    # We should have extracted 10 convolution tasks
    # assert len(tasks) == 10
    print("Extracted {} conv2d tasks:".format(len(tasks)))
    for tsk in tasks:
        inp = tsk.args[0][1]
        wgt = tsk.args[1][1]
        batch = inp[0] * inp[4]
        in_filter = inp[1] * inp[5]
        out_filter = wgt[0] * wgt[4]
        height, width = inp[2], inp[3]
        hkernel, wkernel = wgt[2], wgt[3]
        hstride, wstride = tsk.args[2][0], tsk.args[2][1]
        hpad, wpad = tsk.args[3][0], tsk.args[3][1]
        print("({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(
            batch, height, width, in_filter, out_filter, hkernel, wkernel,
            hpad, wpad, hstride, wstride))

    # We do not run the tuning in our webpage server since it takes too long.
    # Comment the following line to run it by yourself.
    # return

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    # compile kernels with history best records
    with autotvm.tophub.context(target, extra_files=[log_file]):
        # Compile network
        print("Compile...")
        if target.device_name != "vta":
            with tvm.transform.PassContext(opt_level=3, disabled_pass={"AlterOpLayout"}):
                graph, lib, params = relay.build(relay_prog,
                                                target=target,
                                                params=params,
                                                target_host=env.target_host)
        else:
            #print("Dive into relay.build() !!!")
            #pdb.set_trace()

            with vta.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
                graph, lib, params = relay.build(
                    relay_prog,
                    target=target,
                    params=params,
                    target_host=env.target_host)

        # Export library
        print("Upload...")
        temp = util.tempdir()
        lib.save(temp.relpath("graphlib.o"))
        remote.upload(temp.relpath("graphlib.o"))
        lib = remote.load_module("graphlib.o")

        # Generate the graph runtime
        ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)
        m = graph_runtime.create(graph, lib, ctx)

        ######################################################################
        # Perform image classification inference
        # --------------------------------------
        # We run classification on an image sample from ImageNet
        # We just need to download the categories files, `synset.txt`
        # and an input test image.

        # Download ImageNet categories
        categ_url = "https://github.com/uwsaml/web-data/raw/master/vta/models/"
        categ_fn = "synset.txt"
        download.download(join(categ_url, categ_fn), categ_fn)
        synset = eval(open(categ_fn).read())

        # Download test image
        image_url = 'https://homes.cs.washington.edu/~moreau/media/vta/cat.jpg'
        image_fn = 'cat.png'
        download.download(image_url, image_fn)

        # Prepare test image for inference
        image = Image.open(image_fn).resize((224, 224))
        print("meow!")
        plt.imshow(image)
        plt.show()
        image = np.array(image) - np.array([123., 117., 104.])
        image /= np.array([58.395, 57.12, 57.375])
        image = image.transpose((2, 0, 1))
        image = image[np.newaxis, :]
        image = np.repeat(image, env.BATCH, axis=0)
        ######################################################################
        # Set the network parameters and inputs
        m.set_input(**params)
        m.set_input('data', image)

        # Perform inference and gather execution statistics
        # More on: :py:method:`tvm.runtime.Module.time_evaluator`
        num = 4 # number of times we run module for a single measurement
        rep = 3 # number of measurements (we derive std dev from this)
        timer = m.module.time_evaluator("run", ctx, number=num, repeat=rep)

        tcost = timer()
        std = np.std(tcost.results) * 1000
        mean = tcost.mean * 1000
        print("\nPerformed inference in %.2fms (std = %.2f) for %d samples" % (mean, std, env.BATCH))
        print("Average per sample inference time: %.2fms" % (mean/env.BATCH))

        # Get classification results
        tvm_output = m.get_output(0, tvm.nd.empty((env.BATCH, 1000), "float32", remote.cpu(0)))
        for b in range(env.BATCH):
            top_categories = np.argsort(tvm_output.asnumpy()[b])
            # Report top-5 classification results
            print("\n{} prediction for sample {}".format(network, b))
            print("\t#1:", synset[top_categories[-1]])
            print("\t#2:", synset[top_categories[-2]])
            print("\t#3:", synset[top_categories[-3]])
            print("\t#4:", synset[top_categories[-4]])
            print("\t#5:", synset[top_categories[-5]])
            # This just checks that one of the 5 top categories
            # is one variety of cat; this is by no means an accurate
            # assessment of how quantization affects classification
            # accuracy but is meant to catch changes to the
            # quantization pass that would accuracy in the CI.
            cat_detected = False
            for k in top_categories[-5:]:
                if "cat" in synset[k]:
                    cat_detected = True
            assert(cat_detected)

# Run the tuning and evaluate the results
tune_and_evaluate(tuning_option)

