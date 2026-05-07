import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.ops import variable_scope, control_flow_ops, state_ops
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.framework import ops
from collections import OrderedDict
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.layers.python.layers.feature_column_ops import _input_from_feature_columns
from tensorflow.python.ops import math_ops

from model_ops import ops as base_ops
from model_ops import utils, learning_rate_ops
import model_ops.optimizer_ops as myopt
from model_zoo.base_model import BaseModel
from tensorflow.python.ops import nn_ops
from model_ops.tflog import tflogger as logging
from model_ops.attention import multihead_target_attention, feedforward, multihead_attention
from model_ops.efficient_attention import efficient_target_attention
from model_ops.attention_master_wosmax import adaptive_fully_connected, talk_multihead_attention_rms, feedforward_latent_rms, dynamic_custom_rms_norm, silu


# based on YuexiaACRLAfcBase

class YuexiaACRLTransformerBase(BaseModel):
    def __init__(self,
                 model_config,
                 training_config,
                 mc,
                 fg,
                 context,
                 name="CTR"):
        super(YuexiaACRLTransformerBase, self).__init__(model_config,
                                                training_config,
                                                mc,
                                                fg,
                                                context,
                                                name)
        logging.info("[YuexiaACRLTransformerBase|init] model: {}, init start.".format(self.name))

        self.mtl_logits = {}
        self.main_task = self.training_config.main_task
        self.aux_task = self.training_config.aux_task

        self.critic_model_name = "Critic"
        self.ta_num_units = self.training_config.ta_num_units
        self.ta_num_output_units = self.training_config.ta_num_output_units
        self.ta_num_heads = self.training_config.ta_num_heads

        if self.config.main_column_blocks_str != '':
            self.main_column_blocks = self.config.main_column_blocks_str.split(';', -1)
        if self.config.bias_column_blocks_str != '':
            self.bias_column_blocks = self.config.bias_column_blocks_str.split(';', -1)

        # build ta_seq atten
        self.ta_seq_column_blocks = []
        self.ta_seq_block_name_dict = {}
        self.ta_seq_atten_query_dict = {}
        self.ta_seq_atten_num_units = {}
        self.ta_seq_atten_type_dict = {}
        self.ta_seq_ffn_dict = {}
        self.update_column2emb_dict = {}

        if self.config.ta_seq_column_blocks_str != '':
            arr_blocks = self.config.ta_seq_column_blocks_str.split(';', -1)
            for block in arr_blocks:
                arr = block.split(':', -1)
                seq_name = arr[0].strip()
                ta_query = arr[1].strip()
                unit_num = int(arr[2].strip())
                atten_type = arr[3].strip()
                ffn = bool(int(arr[4].strip()))

                block_name = seq_name + '_' + atten_type  # TODO: modify
                self.ta_seq_block_name_dict[seq_name] = block_name
                self.ta_seq_column_blocks.append(seq_name)
                self.ta_seq_atten_query_dict[seq_name] = ta_query
                self.ta_seq_atten_num_units[seq_name] = unit_num
                self.ta_seq_atten_type_dict[seq_name] = atten_type
                self.ta_seq_ffn_dict[seq_name] = ffn

                if seq_name not in self.main_column_blocks:
                    self.main_column_blocks.append(seq_name)
                if block_name not in self.main_column_blocks:
                    self.main_column_blocks.append(block_name)
                if ta_query not in self.main_column_blocks:
                    self.main_column_blocks.append(ta_query)

        self.token_project_feat = 'CTR_token_project_feat'
        self.token_project_feat_output = 'CTR_token_project_feat_output'
        self.token_project_feat_seq = 'CTR_token_project_feat'
        self.token_project_feat_output_seq = 'CTR_token_project_feat_output'
        self.satten_collections_transformer_layer = "satten_transformer_layer"
        self.satten_collections_transformer_output = "satten_transformer_output"

    def embedding_layer(self):
        logging.info("[YuexiaACRLTransformerBase|embedding_layer] model: {}, embedding_layer start.".format(self.name))
        super(YuexiaACRLTransformerBase, self).embedding_layer()
        with tf.variable_scope(name_or_scope="seq_input_from_feature_columns",
                               partitioner=base_ops.partitioner(self.config.ps_num,
                                                                self.training_config.embedding_partition_size),
                               reuse=tf.AUTO_REUSE) as scope:
            logging.info(
                "[YuexiaACRLTransformerBase|seq_embedding_layer] model: {}, self.seq_column_blocks: {}".format(self.name,
                                                                                                       str(self.seq_column_blocks)))
            for block_name in self.seq_column_blocks:
                logging.info(
                    "[YuexiaACRLTransformerBase|seq_embedding_layer] model: {}, block_name: {}".format(self.name, block_name))
                if (not self.mc.seq_has_block(block_name) or
                        len(self.mc.get_seq_column_names_by_block_name(block_name)) == 0):
                    logging.warn('[Fetch embedding, seq mc conf] does not has block : {}'.format(block_name))
                    continue
                logging.info(
                    "[YuexiaACRLTransformerBase|seq_embedding_layer] model: {}, seq_column_blocks, block_name: {}".format(
                        self.name, block_name))

                # feature_columns[block_name]: [f_num, seq_length*N, f_embedding]
                # sequence_layer: [seq_length*N, f_num*f_embedding]
                if self.config.fix_sorted_columns:
                    block_columns = [block_name + '_' + column for column in
                                     self.mc.get_seq_column_names_by_block_name(block_name)]
                    block_feature_columns = self.feature_columns[block_name]
                    if self.config.is_columnize:
                        self.block_columns_embedding_dict[block_name] = OrderedDict([(column_name,
                                                                                      _input_from_feature_columns(
                                                                                          columns_to_tensors=self.features,
                                                                                          feature_columns=[
                                                                                              feature_column],
                                                                                          weight_collections=None,
                                                                                          trainable=True,
                                                                                          scope=scope,
                                                                                          output_rank=3,
                                                                                          default_name='seq_input_from_feature_columns'))
                                                                                     for column_name, feature_column in
                                                                                     zip(block_columns,
                                                                                         block_feature_columns)])
                    else:
                        self.block_columns_embedding_dict[block_name] = OrderedDict([(column_name,
                                                                                      layers.input_from_feature_columns(
                                                                                          columns_to_tensors=self.features,
                                                                                          feature_columns=[
                                                                                              feature_column],
                                                                                          scope=scope))
                                                                                     for column_name, feature_column in
                                                                                     zip(block_columns,
                                                                                         block_feature_columns)])
                    columns_embedding = [self.block_columns_embedding_dict[block_name][column_name] for column_name in
                                         block_columns]
                    sequence_layer = tf.concat(columns_embedding, axis=-1)
                else:
                    if self.config.is_columnize:
                        sequence_layer = _input_from_feature_columns(
                            columns_to_tensors=self.features,
                            feature_columns=self.feature_columns[block_name],
                            weight_collections=None,
                            trainable=True,
                            scope=scope,
                            output_rank=3,
                            default_name='seq_input_from_feature_columns'
                        )
                    else:
                        sequence_layer = layers.input_from_feature_columns(
                            self.features,
                            self.feature_columns[block_name],
                            scope=scope)
                logging.info(
                    "[YuexiaACRLTransformerBase|seq_embedding_layer] model: {}, seq_column_blocks, block_name: {}, sequence_layer shape: {}".format(
                        self.name, block_name, sequence_layer.get_shape().as_list()))

                if self.config.is_columnize:
                    seq_len = self.fg.get_seq_len_by_sequence_name(block_name)
                    sequence_stack = tf.reshape(sequence_layer, [-1, seq_len, sequence_layer.get_shape().as_list()[-1]])
                else:
                    seq_len = self.fg.get_seq_len_by_sequence_name(block_name)
                    sequence = tf.split(sequence_layer, seq_len, axis=0)  # [seq_length, N, f_num*f_embedding]
                    sequence_stack = tf.stack(values=sequence, axis=1)  # [N, seq_length, f_num*f_embedding]
                logging.info(
                    "[YuexiaACRLTransformerBase|seq_embedding_layer] model: {}, seq_column_blocks, block_name: {}, sequence_stack shape: {}".format(
                        self.name, block_name, sequence_stack.get_shape().as_list()))

                seq_real_length_name = '{}_length'.format(block_name)
                logging.info(
                    "[YuexiaACRLTransformerBase|seq_embedding_layer] model: {}, seq_real_length_name: {}".format(self.name,
                                                                                                         seq_real_length_name))

                seq_real_length = self.block_layer_dict[seq_real_length_name]  # [N, 1]
                sequence_mask = tf.sequence_mask(tf.reshape(seq_real_length, [-1]), seq_len)  # [N, seq_length]

                sequence_2d = tf.reshape(sequence_stack,
                                         [-1, tf.shape(sequence_stack)[2]])  # [N*seq_length, f_num*f_embedding]
                sequence_stack = tf.reshape(tf.where(tf.reshape(sequence_mask, [-1]),
                                                     sequence_2d, tf.zeros_like(sequence_2d)),
                                            tf.shape(sequence_stack))  # [N, seq_length, f_num*f_embedding]
                self.seq_block_layer_dict[block_name] = sequence_stack  # [N, seq_length, f_num*f_embedding]
                logging.info(
                    "[YuexiaACRLTransformerBase|seq_embedding_layer] model: {}, seq_column_blocks, block_name: {}, sequence_final shape: {}".format(
                        self.name, block_name, self.seq_block_layer_dict[block_name].get_shape().as_list()))

    def main_net(self):
        logging.info("[YuexiaACRLTransformerBase|main_net] model: {}, main_net start.".format(self.name))
        with tf.variable_scope(name_or_scope="{}_Main_Net".format(self.critic_model_name),
                               partitioner=base_ops.partitioner(self.config.ps_num,
                                                                self.training_config.dnn_partition_size),
                               reuse=tf.AUTO_REUSE):
            is_training = self.is_training if self.name == self.critic_model_name else False

            main_net_layer = []
            for block_name in self.main_column_blocks:
                if block_name not in self.block_layer_dict:
                    logging.warn(
                        '[YuexiaACRLTransformerBase|main_net] block_layer_dict and update_column2emb_dict does not has block : {}'.format(
                            block_name))
                    continue
                logging.info(
                    "[YuexiaACRLTransformerBase|main_net] model: {}, block_name: {} add to main_net_layer".format(self.name,
                                                                                                          block_name))

                if self.name != self.critic_model_name and block_name in self.update_column2emb_dict:
                    add_feat = self.update_column2emb_dict[block_name]
                else:
                    add_feat = self.block_layer_dict[block_name]
                logging.info(
                    "[YuexiaACRLTransformerBase|main_net] model: {}, block_name: {}, add_feat shape: {}".format(self.name,
                                                                                                        block_name,
                                                                                                        add_feat.get_shape().as_list()))
                if self.config.use_main_net_add_feat:
                    main_feat = self.main_net_add_feat(block_name,
                                                       add_feat,
                                                       self.config.dnn_hidden_units[0],
                                                       is_training)
                    main_net_layer.append(main_feat)
                else:
                    main_net_layer.append(add_feat)
            if self.config.use_main_net_add_feat:
                main_net_input = tf.add_n(main_net_layer)
            else:
                main_net_input = tf.concat(values=main_net_layer, axis=1)
            logging.info("[YuexiaACRLTransformerBase|main_net] model: {}, main_net_input shape: {}".format(self.name,
                                                                                                   main_net_input.get_shape().as_list()))

            for taskName in self.mtl_task:
                self.mtl_main_net[taskName] = main_net_input

            with arg_scope(base_ops.model_arg_scope(weight_decay=self.training_config.dnn_l2_reg)):
                for layer_id, num_hidden_units in enumerate(self.config.dnn_hidden_units):
                    for taskName in self.mtl_task:
                        with variable_scope.variable_scope(
                                "hiddenlayer_{}_{}".format(taskName, layer_id)) as dnn_hidden_layer_scope:
                            self.mtl_main_net[taskName] = adaptive_fully_connected(
                                self.mtl_main_net[taskName],
                                self.latent_condition,
                                num_hidden_units,
                                utils.getActivationFunctionOp(self.config.activation_op),
                                scope=dnn_hidden_layer_scope,
                                variables_collections=[self.collections_dnn_hidden_layer],
                                outputs_collections=[self.collections_dnn_hidden_output],
                                normalizer_fn=layers.batch_norm,
                                normalizer_params={"scale": True, "is_training": is_training,
                                                   "fused": False} if self.training_config.use_gpu
                                else {"scale": True, "is_training": is_training})
            self.main_net_ = self.mtl_main_net[self.clkTaskName]

    def main_net_add_feat(self, index, inputs, num_hidden_units, is_training):
        with variable_scope.variable_scope("main_net_add_feat_{}".format(index),
                                           reuse=tf.AUTO_REUSE) as scope:
            outputs = adaptive_fully_connected(
                inputs,
                self.latent_condition,
                num_hidden_units,
                utils.getActivationFunctionOp(self.config.activation_op),
                scope=scope,
                variables_collections=[self.collections_dnn_hidden_layer],
                outputs_collections=[self.collections_dnn_hidden_output],
                normalizer_fn=layers.batch_norm,
                normalizer_params={"scale": True, "is_training": is_training,
                                   "fused": False} if self.training_config.use_gpu
                else {"scale": True, "is_training": is_training})
        return outputs

    def bias_net(self):
        logging.info("[YuexiaACRLTransformerBase|bias_net] model: {}, bias_net start.".format(self.name))
        if not self.config.use_bias_net:
            return

        with tf.variable_scope(name_or_scope="{}_Bias_Net".format(self.critic_model_name),
                               partitioner=base_ops.partitioner(self.config.ps_num,
                                                                self.training_config.dnn_partition_size),
                               reuse=tf.AUTO_REUSE):
            is_training = self.is_training if self.name == self.critic_model_name else False

            bias_net_layer = []
            for block_name in self.bias_column_blocks:
                if block_name not in self.block_layer_dict:
                    logging.warn('[Bias net, block layer dict] does not has block : {}'.format(block_name))
                    continue
                bias_net_layer.append(self.block_layer_dict[block_name])
                logging.info(
                    "[YuexiaACRLTransformerBase|bias_net] model: {}, block_name: {} add to bias_net_layer".format(self.name,
                                                                                                          block_name))
            bias_net_input = tf.concat(values=bias_net_layer, axis=1)
            logging.info("[YuexiaACRLTransformerBase|bias_net] model: {}, bias_net_input shape: {}".format(self.name,
                                                                                                   bias_net_input.get_shape().as_list()))

            self.bias_net_ = bias_net_input

            with arg_scope(base_ops.model_arg_scope(weight_decay=self.training_config.dnn_l2_reg)):
                for layer_id, num_hidden_units in enumerate(self.config.bias_dnn_hidden_units):
                    with variable_scope.variable_scope(
                            "hiddenlayer_{}".format(layer_id)) as dnn_hidden_layer_scope:
                        self.bias_net_ = adaptive_fully_connected(
                            self.bias_net_,
                            self.latent_condition,
                            num_hidden_units,
                            utils.getActivationFunctionOp(self.config.activation_op),
                            scope=dnn_hidden_layer_scope,
                            variables_collections=[self.collections_dnn_hidden_layer],
                            outputs_collections=[self.collections_dnn_hidden_output],
                            normalizer_fn=layers.batch_norm,
                            normalizer_params={"scale": True, "is_training": is_training,
                                               "fused": False} if self.training_config.use_gpu
                            else {"scale": True, "is_training": is_training})

            logging.info("[YuexiaACRLTransformerBase|bias_net] model: {}, bias_net_output shape: {}".format(self.name,
                                                                                                    self.bias_net_.get_shape().as_list()))

    def logits_layer(self):
        logging.info("[YuexiaACRLTransformerBase|logits_layer] model: {}, logits_layer start.".format(self.name))
        with tf.variable_scope(name_or_scope="{}_Logits".format(self.critic_model_name),
                               partitioner=base_ops.partitioner(self.config.ps_num,
                                                                self.training_config.dnn_partition_size),
                               reuse=tf.AUTO_REUSE) as dnn_logits_scope:

            with arg_scope(base_ops.model_arg_scope(weight_decay=self.training_config.dnn_l2_reg)):
                if self.config.use_bias_net:
                    self.bias_logits = layers.linear(
                        self.bias_net_,
                        1,
                        scope="bias_net_logits",
                        variables_collections=[self.collections_dnn_hidden_layer],
                        outputs_collections=[self.collections_dnn_hidden_output],
                        biases_initializer=None)
                self.bias = contrib_variables.model_variable(
                    'bias_weight',
                    shape=[1],
                    initializer=tf.zeros_initializer(),
                    trainable=True)
                for taskName in self.mtl_task:
                    if taskName in self.main_task:
                        self.mtl_main_logits[taskName] = layers.linear(
                            self.transformer_output,
                            1,
                            scope="main_net_{}".format(taskName),
                            variables_collections=[self.collections_dnn_hidden_layer],
                            outputs_collections=[self.collections_dnn_hidden_output],
                            biases_initializer=None)
                    else:
                        self.mtl_main_logits[taskName] = layers.linear(
                            self.transformer_output,
                            1,
                            scope="main_net_{}".format(taskName),
                            variables_collections=[self.collections_dnn_hidden_layer],
                            outputs_collections=[self.collections_dnn_hidden_output],
                            biases_initializer=None)
                    if self.config.use_bias_net:
                        self.mtl_logits[taskName] = nn_ops.bias_add(self.mtl_main_logits[taskName] + self.bias_logits,
                                                                    self.bias)
                    else:
                        self.mtl_logits[taskName] = nn_ops.bias_add(self.mtl_main_logits[taskName], self.bias)

            self.logits = self.mtl_logits[self.clkTaskName]

    def seq_pooling_layer(self):
        logging.info("[YuexiaACRLTransformerBase|seq_pooling_layer] model: {}, seq_pooling_layer start.".format(self.name))
        for seq_name in self.ta_seq_column_blocks:
            if seq_name not in self.seq_block_layer_dict:
                logging.warn('[Seq block layer dict] does not has block : {}'.format(seq_name))
                continue
            logging.info(
                "[YuexiaACRLTransformerBase|seq_pooling_layer] model_name: {}, seq_name: {}".format(self.name, seq_name))
            with arg_scope(base_ops.model_arg_scope(weight_decay=self.training_config.dnn_l2_reg)):
                with tf.variable_scope(name_or_scope=seq_name + "_Pooling_Layer",
                                       partitioner=base_ops.partitioner(self.config.ps_num,
                                                                        self.training_config.dnn_partition_size),
                                       reuse=tf.AUTO_REUSE) as scope:
                    sequence = self.seq_block_layer_dict[seq_name]
                    sequence = layers.fully_connected(sequence,
                                                      num_outputs=sequence.get_shape().as_list()[-1],
                                                      activation_fn=utils.getActivationFunctionOp(
                                                          self.config.activation_op),
                                                      scope=scope,
                                                      reuse=tf.AUTO_REUSE,
                                                      variables_collections="{}_pooling_dnn_hidden_layer".format(
                                                          seq_name),
                                                      outputs_collections="{}_pooling_dnn_hidden_output".format(
                                                          seq_name))
                    self.block_layer_dict[seq_name] = tf.reduce_mean(sequence, axis=1)  # [N, D]
                    logging.info(
                        "[YuexiaACRLTransformerBase|seq_pooling_layer] {} shape: {}".format(seq_name, self.block_layer_dict[
                            seq_name].get_shape().as_list()))

    def seq_target_atten_layer(self):
        logging.info(
            "[YuexiaACRLTransformerBase|seq_target_atten_layer] model: {}, seq_target_atten_layer start.".format(self.name))
        for seq_name in self.ta_seq_column_blocks:
            block_name = self.ta_seq_block_name_dict[seq_name]
            num_units = self.ta_seq_atten_num_units[seq_name]
            atten_type = self.ta_seq_atten_type_dict[seq_name]
            ta_query = self.ta_seq_atten_query_dict[seq_name]
            use_ffn = self, self.ta_seq_ffn_dict[seq_name]
            if self.name == self.critic_model_name:
                atten_query = self.block_layer_dict[ta_query]
            else:
                atten_query = self.update_column2emb_dict[ta_query]
            atten_query = tf.expand_dims(atten_query, 1)

            with arg_scope(base_ops.model_arg_scope(weight_decay=self.training_config.dnn_l2_reg)):
                with tf.variable_scope(
                        name_or_scope=self.critic_model_name + "_" + block_name + "_Target_Attention_Layer",
                        partitioner=base_ops.partitioner(self.config.ps_num,
                                                         self.training_config.dnn_partition_size),
                        reuse=tf.AUTO_REUSE) as scope:
                    max_len = self.fg.get_seq_len_by_sequence_name(seq_name)
                    sequence_length = self.block_layer_dict[seq_name + '_length']
                    sequence_mask = tf.sequence_mask(tf.reshape(sequence_length, [-1]), max_len)  # [N, 1024]
                    self.seq_real_len[seq_name] = tf.reduce_sum(tf.cast(sequence_mask, tf.float32), axis=1)
                    atten_key = self.seq_block_layer_dict[seq_name]
                    atten_value = self.seq_block_layer_dict[seq_name]
                    logging.info(
                        "[YuexiaACRLTransformerBase|seq_target_atten_layer] model: {}, seq_name: {}, atten_query shape: {}, atten_key shape: {}, atten_value shape: {}.".format(
                            self.name,
                            seq_name,
                            atten_query.get_shape().as_list(),
                            atten_key.get_shape().as_list(),
                            atten_value.get_shape().as_list()
                        ))

                    if atten_type.lower() == 'eta':
                        logging.info(
                            "[YuexiaACRLTransformerBase|seq_target_atten_layer] model: {}, seq_name: {}, use eta.".format(
                                self.name, seq_name))
                        ua_item_vec, uatt_vec = efficient_target_attention(queries=atten_query,
                                                                           keys=atten_key,
                                                                           values=atten_value,
                                                                           num_units=num_units,
                                                                           num_output_units=num_units,
                                                                           activation_fn=None,
                                                                           scope="efficient_target_attention",
                                                                           reuse=tf.AUTO_REUSE,
                                                                           key_masks=sequence_mask,
                                                                           atten_mode=self.config.long_seq_ta_atten_mode,
                                                                           linear_projection=self.config.long_seq_ta_linear_projection,
                                                                           variables_collections=[
                                                                               "{}_atten_dnn_hidden_layer".format(
                                                                                   seq_name)],
                                                                           outputs_collections=[
                                                                               "{}_atten_dnn_hidden_output".format(
                                                                                   seq_name)],
                                                                           num_heads=self.config.long_seq_num_heads,
                                                                           topk=self.config.long_seq_topk,
                                                                           n_hashes=self.config.long_seq_n_hashes)
                    elif atten_type.lower() == 'target':
                        logging.info(
                            "[YuexiaACRLTransformerBase|seq_target_atten_layer] model: {}, seq_name: {}, use target.".format(
                                self.name,
                                seq_name))
                        ua_item_vec, uatt_vec = multihead_target_attention(queries=atten_query,
                                                                           keys=atten_key,
                                                                           values=atten_value,
                                                                           num_units=num_units,
                                                                           num_output_units=num_units,
                                                                           activation_fn=None,
                                                                           scope="target_attention",
                                                                           reuse=tf.AUTO_REUSE,
                                                                           key_masks=sequence_mask,
                                                                           atten_mode=self.config.atten_mode,
                                                                           linear_projection=self.config.ta_linear_projection,
                                                                           fix_rtp_bug=self.config.ta_fix_rtp_bug,
                                                                           variables_collections=[
                                                                               "{}_atten_dnn_hidden_layer".format(
                                                                                   seq_name)],
                                                                           outputs_collections=[
                                                                               "{}_atten_dnn_hidden_output".format(
                                                                                   seq_name)],
                                                                           num_heads=self.ta_num_heads)
                    elif atten_type.lower() == 'poly':
                        logging.info(
                            "[YuexiaACRLTransformerBase|seq_target_atten_layer] model: {}, seq_name: {}, use poly-encoder.".format(
                                self.name, seq_name))

                        key_seq = self.seq_block_layer_dict[seq_name]
                        key_mask = sequence_mask
                        #### query
                        poly_matrix_long = tf.get_variable(
                            name="poly_anchor_union_seq",
                            shape=[self.training_config.ncodes, self.training_config.ncodes_dim],  # [16, 32]
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(128 ** (-0.5)),
                            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.MODEL_VARIABLES]
                        )
                        #### metric
                        poly_norm_long = tf.nn.l2_normalize(poly_matrix_long, axis=1)  # shape=(code, fdim)
                        poly_cov_long = tf.matmul(poly_norm_long, tf.transpose(poly_norm_long))  # shape=(code, code)
                        poly_cov_long = (tf.ones_like(poly_cov_long) - tf.eye(
                            self.training_config.ncodes)) * poly_cov_long  # shape=(code, code)
                        self.cov_loss_long = tf.reduce_sum(tf.abs(poly_cov_long))
                        ##### Attention Query Build
                        poly_matrix_long_tile = tf.tile(tf.expand_dims(poly_matrix_long, 0),
                                                        [tf.shape(key_seq)[0], 1, 1])  # shape=(batch, code, fdim)
                        query_seq = poly_matrix_long_tile
                        query_mask = tf.sequence_mask(
                            self.training_config.ncodes * tf.ones_like(query_seq[:, 0, 0], dtype=tf.int32),
                            self.training_config.ncodes
                        ),
                        sequence, stt_vec = multihead_attention(queries=query_seq,
                                                                keys=key_seq,
                                                                num_units=self.config.sa_num_units,
                                                                num_output_units=self.config.sa_num_output_units,
                                                                activation_fn=None,
                                                                scope="act_attention_1",
                                                                reuse=tf.AUTO_REUSE,
                                                                query_masks=query_mask,
                                                                key_masks=key_mask,
                                                                atten_mode=self.config.atten_mode,
                                                                linear_projection=self.config.sa_linear_projection,
                                                                fix_rtp_bug=self.config.ta_fix_rtp_bug,
                                                                variables_collections=["act_attention_1_layer"],
                                                                outputs_collections=["act_attention_1_output"],
                                                                num_heads=self.config.num_heads)
                        tf.summary.histogram('[{}_act_layer_1_stt_vec]'.format(seq_name), stt_vec)

                        # atten_query = self.block_layer_dict[act_query]  # [N, D]
                        # atten_query = tf.expand_dims(atten_query, 1)  # [N, 1, D]
                        atten_key = sequence
                        atten_value = sequence
                        # ua_item_vec: [N, 1, 128]
                        ua_item_vec, uatt_vec = multihead_target_attention(queries=atten_query,
                                                                           keys=atten_key,
                                                                           values=atten_value,
                                                                           num_units=num_units,
                                                                           num_output_units=num_units,
                                                                           activation_fn=None,
                                                                           scope="act_attention_2",
                                                                           reuse=tf.AUTO_REUSE,
                                                                           key_masks=query_mask,
                                                                           atten_mode=self.config.atten_mode,
                                                                           linear_projection=self.config.ta_linear_projection,
                                                                           fix_rtp_bug=self.config.ta_fix_rtp_bug,
                                                                           variables_collections=[
                                                                               "act_attention_2_layer"],
                                                                           outputs_collections=[
                                                                               "act_attention_2_output"],
                                                                           num_heads=self.config.num_heads)
                        tf.summary.histogram('{}_act_layer_2_stt_vec'.format(seq_name), uatt_vec)
                    else:
                        logging.info(
                            "[YuexiaACRLTransformerBase|seq_target_atten_layer] model: {}, seq_name: {}, use target.".format(
                                self.name,
                                seq_name))
                        ua_item_vec, uatt_vec = multihead_target_attention(queries=atten_query,
                                                                           keys=atten_key,
                                                                           values=atten_value,
                                                                           num_units=num_units,
                                                                           num_output_units=num_units,
                                                                           activation_fn=None,
                                                                           scope="target_attention",
                                                                           reuse=tf.AUTO_REUSE,
                                                                           key_masks=sequence_mask,
                                                                           atten_mode=self.config.atten_mode,
                                                                           linear_projection=self.config.ta_linear_projection,
                                                                           fix_rtp_bug=self.config.ta_fix_rtp_bug,
                                                                           variables_collections=[
                                                                               "{}_atten_dnn_hidden_layer".format(
                                                                                   seq_name)],
                                                                           outputs_collections=[
                                                                               "{}_atten_dnn_hidden_output".format(
                                                                                   seq_name)],
                                                                           num_heads=self.ta_num_heads)

                    if use_ffn:
                        item_vec = feedforward(ua_item_vec,
                                               num_units=[num_units, num_units],
                                               activation_fn=utils.getActivationFunctionOp(
                                                   self.config.activation_op),
                                               scope="ua_feed_forward",
                                               reuse=tf.AUTO_REUSE,
                                               variables_collections=["{}_atten_dnn_hidden_layer_ffn".format(seq_name)],
                                               outputs_collections=["{}_atten_dnn_hidden_output_ffn".format(seq_name)])
                    else:
                        item_vec = ua_item_vec

                    dec = tf.reshape(item_vec, [-1, num_units])  # [N, 128]
                    logging.info(
                        "[YuexiaACRLTransformerBase|seq_target_atten_layer] model: {}, seq: {}, atten_type: {}, shape: {}.".format(
                            self.name, seq_name,
                            atten_type,
                            dec.get_shape().as_list()))

                self.block_layer_dict[block_name] = dec
                logging.info(
                    "[YuexiaACRLTransformerBase|seq_target_atten_layer] model: {}, seq: {}, final shape: {}".format(self.name,
                                                                                                            seq_name,
                                                                                                            self.block_layer_dict[
                                                                                                                block_name].get_shape().as_list()))

    def latent_condition_transform(self):
        logging.info("[YuexiaACRLTransformerBase|seq_target_atten_layer] model: {}, Use latent_condition_transform layer.".format(self.name))
        with arg_scope(base_ops.model_arg_scope(weight_decay=self.training_config.dnn_l2_reg)):
            with tf.variable_scope(name_or_scope="Latent_Condition_Transform_Layer",
                                   partitioner=base_ops.partitioner(self.config.ps_num,
                                                                    self.training_config.dnn_partition_size),
                                   reuse=tf.AUTO_REUSE) as scope:

                if self.name == self.critic_model_name:
                    latent_condition = self.block_layer_dict["item_columns"]
                    account_latent_condition = self.block_layer_dict["account_columns"]
                else:
                    latent_condition = self.update_column2emb_dict["critic_item_columns"]
                    account_latent_condition = self.block_layer_dict["critic_account_columns"]

                if self.config.use_account_latent_cond:
                    latent_condition = tf.concat([account_latent_condition, latent_condition], axis=-1)

                self.latent_condition = layers.fully_connected(latent_condition,
                                                               num_outputs=256,
                                                               activation_fn=utils.getActivationFunctionOp(
                                                                   self.config.activation_op),
                                                               scope=scope,
                                                               reuse=tf.AUTO_REUSE,
                                                               variables_collections=self.collections_dnn_hidden_layer,
                                                               outputs_collections=self.collections_dnn_hidden_output)
                logging.info("[YuexiaACRLTransformerBase|seq_target_atten_layer] latent_condition output shape: {}".format(
                    self.latent_condition.get_shape().as_list()))
        logging.info("[YuexiaACRLTransformerBase|seq_target_atten_layer] latent_condition_transform layer end.")

    def token_project_feat_layer(self, tag, index, inputs, num_hidden_units, variables_collections, outputs_collections):
        with variable_scope.variable_scope("token_project_feat_layer_{}_{}".format(tag, index),
                                           reuse=tf.AUTO_REUSE) as scope:
            outputs = layers.fully_connected(
                inputs,
                num_hidden_units,
                activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.00001),
                biases_initializer=None,
                scope=scope,
                variables_collections=[variables_collections],
                outputs_collections=[outputs_collections]
            )
        return outputs

    def transformer_layer(self):
        logging.info("[YuexiaACRLTransformerBase|transformer_layer] use token transformer_layer")
        is_training = self.is_training if self.name == self.critic_model_name else False

        with arg_scope(base_ops.model_arg_scope(weight_decay=self.training_config.dnn_l2_reg)):
            with tf.variable_scope(name_or_scope="Token_Transformer_Layer",
                                   partitioner=base_ops.partitioner(self.config.ps_num,
                                                                    self.training_config.dnn_partition_size),
                                   reuse=tf.AUTO_REUSE) as scope:

                total_tokens = []
                total_tokens_mask = []
                for index, block_name in enumerate(self.main_column_blocks):
                    if block_name not in self.block_layer_dict:
                        logging.warn('[YuexiaACRLTransformerBase|transformer_layer] block_layer_dict and update_column2emb_dict does not has block : {}'.format(block_name))
                        continue

                    if self.name != self.critic_model_name and block_name in self.update_column2emb_dict:
                        add_feat = self.update_column2emb_dict[block_name]
                    else:
                        add_feat = self.block_layer_dict[block_name]

                    project_feat = self.token_project_feat_layer('scale', block_name, add_feat,
                                                                 self.config.transformer_units,
                                                                 self.token_project_feat,
                                                                 self.token_project_feat_output)
                    project_feat = tf.expand_dims(project_feat, 1)
                    total_tokens.append(project_feat)
                    logging.info("[YuexiaACRLTransformerBase|transformer_layer] index: {}, key: {}, in shape: {}, output shape: {}".format(
                            index, block_name, add_feat.get_shape().as_list(), project_feat.get_shape().as_list()))

                    if block_name.endswith('_seq'):
                        seq_len = 1
                        seq_real_length_name = '{}_length'.format(block_name)
                        seq_real_length = self.block_layer_dict[seq_real_length_name]  # [N, 1]
                        seq_real_length = tf.ones_like(seq_real_length)
                        sequence_mask = tf.sequence_mask(tf.reshape(seq_real_length, [-1]), seq_len)  # [N, seq_length]
                        total_tokens_mask.append(sequence_mask)
                    else:
                        shape = tf.shape(project_feat)[:2]
                        mask = tf.ones(shape, dtype=tf.bool)
                        total_tokens_mask.append(mask)

                self.token_seq_len = len(total_tokens)
                logging.info("[YuexiaACRLTransformerBase|transformer_layer] token len: {}".format(self.token_seq_len))

                if self.training_config.use_gpu:
                    with tf.device('/CPU:0'):
                        self.main_token_feature = tf.concat(values=total_tokens, axis=1)
                        self.main_token_feature_mask = tf.concat(values=total_tokens_mask, axis=1)
                else:
                    self.main_token_feature = tf.concat(values=total_tokens, axis=1)
                    self.main_token_feature_mask = tf.concat(values=total_tokens_mask, axis=1)
                logging.info(
                    "[YuexiaACRLTransformerBase|transformer_layer] main_token_feature shape: {}, mask shape: {}".format(
                        self.main_token_feature.get_shape().as_list(),
                        self.main_token_feature_mask.get_shape().as_list()))

                attn_out1, stt_vec_1, attn_mv1, attn_pv1 = talk_multihead_attention_rms(queries=self.main_token_feature,
                                                                                        keys=self.main_token_feature,
                                                                                        num_units=self.config.transformer_units,
                                                                                        num_output_units=self.config.transformer_units,
                                                                                        activation_fn=None,
                                                                                        scope="self_attention_1",
                                                                                        reuse=tf.AUTO_REUSE,
                                                                                        query_masks=self.main_token_feature_mask,
                                                                                        key_masks=self.main_token_feature_mask,
                                                                                        atten_mode=self.config.atten_mode,
                                                                                        linear_projection=False,
                                                                                        fix_rtp_bug=self.config.ta_fix_rtp_bug,
                                                                                        variables_collections=[
                                                                                            self.satten_collections_transformer_layer],
                                                                                        outputs_collections=[
                                                                                            self.satten_collections_transformer_output],
                                                                                        num_heads=self.config.num_heads,
                                                                                        seq_len_max_Q=self.token_seq_len,
                                                                                        seq_len_max_KV=self.token_seq_len)
                logging.info(
                    "[YuexiaACRLTransformerBase|transformer_layer] attn_out1 shape: {}, stt_vec_1 shape: {}, attn_mv1: {}, attn_pv1: {}".format(
                        attn_out1.get_shape().as_list(), stt_vec_1.get_shape().as_list(), attn_mv1, attn_pv1))

                attn_out1 += self.main_token_feature
                ff_out1, ff_mv_1, ff_pv_1 = feedforward_latent_rms(inputs=attn_out1,
                                                                   cond_feat=self.latent_condition,
                                                                   num_units=[self.config.transformer_units * 4,
                                                                              self.config.transformer_units],
                                                                   activation_fn=silu,
                                                                   scope="feedforward_1",
                                                                   reuse=tf.AUTO_REUSE,
                                                                   variables_collections=[
                                                                       self.satten_collections_transformer_layer],
                                                                   outputs_collections=[
                                                                       self.satten_collections_transformer_output],
                                                                   seq_len_max=self.token_seq_len,
                                                                   hidden_dim=self.config.transformer_units)
                logging.info(
                    "[YuexiaACRLTransformerBase|transformer_layer] ff_out1 shape: {}, ff_mv_1: {}, ff_pv_1: {}".format(
                        ff_out1.get_shape().as_list(), ff_mv_1, ff_pv_1))
                ff_out1 += attn_out1

                self.attn_embedding_table_g_var = attn_pv1[0][0]
                self.attn_embedding_table_b_var = attn_pv1[0][1]
                self.ffn_embedding_table_g_var = ff_pv_1[0][0]
                self.ffn_embedding_table_b_var = ff_pv_1[0][1]

                attn_out2, stt_vec_2, attn_mv2, attn_pv2 = talk_multihead_attention_rms(queries=ff_out1,
                                                                                        keys=ff_out1,
                                                                                        num_units=self.config.transformer_units,
                                                                                        num_output_units=self.config.transformer_units,
                                                                                        activation_fn=None,
                                                                                        scope="self_attention_2",
                                                                                        reuse=tf.AUTO_REUSE,
                                                                                        query_masks=self.main_token_feature_mask,
                                                                                        key_masks=self.main_token_feature_mask,
                                                                                        atten_mode=self.config.atten_mode,
                                                                                        linear_projection=False,
                                                                                        fix_rtp_bug=self.config.ta_fix_rtp_bug,
                                                                                        variables_collections=[
                                                                                            self.satten_collections_transformer_layer],
                                                                                        outputs_collections=[
                                                                                            self.satten_collections_transformer_output],
                                                                                        num_heads=self.config.num_heads,
                                                                                        seq_len_max_Q=self.token_seq_len,
                                                                                        seq_len_max_KV=self.token_seq_len)
                logging.info(
                    "[YuexiaACRLTransformerBase|transformer_layer] attn_out2 shape: {}, stt_vec_2 shape: {}, attn_mv2: {}, attn_pv2: {}".format(
                        attn_out2.get_shape().as_list(), stt_vec_2.get_shape().as_list(), attn_mv2, attn_pv2))

                attn_out2 += ff_out1
                ff_out2, ff_mv_2, ff_pv_2 = feedforward_latent_rms(inputs=attn_out2,
                                                                   cond_feat=self.latent_condition,
                                                                   num_units=[self.config.transformer_units * 4,
                                                                              self.config.transformer_units],
                                                                   activation_fn=silu,
                                                                   scope="feedforward_2",
                                                                   reuse=tf.AUTO_REUSE,
                                                                   variables_collections=[
                                                                       self.satten_collections_transformer_layer],
                                                                   outputs_collections=[
                                                                       self.satten_collections_transformer_output],
                                                                   seq_len_max=self.token_seq_len,
                                                                   hidden_dim=self.config.transformer_units)
                logging.info(
                    "[YuexiaACRLTransformerBase|transformer_layer] ff_out2 shape: {}, ff_mv_2: {}, ff_pv_2: {}".format(
                        ff_out2.get_shape().as_list(), ff_mv_2, ff_pv_2))
                ff_out2 += attn_out2

                attn_out3, stt_vec_3, attn_mv3, attn_pv3 = talk_multihead_attention_rms(queries=ff_out2,
                                                                                        keys=ff_out2,
                                                                                        num_units=self.config.transformer_units,
                                                                                        num_output_units=self.config.transformer_units,
                                                                                        activation_fn=None,
                                                                                        scope="self_attention_3",
                                                                                        reuse=tf.AUTO_REUSE,
                                                                                        query_masks=self.main_token_feature_mask,
                                                                                        key_masks=self.main_token_feature_mask,
                                                                                        atten_mode=self.config.atten_mode,
                                                                                        linear_projection=False,
                                                                                        fix_rtp_bug=self.config.ta_fix_rtp_bug,
                                                                                        variables_collections=[
                                                                                            self.satten_collections_transformer_layer],
                                                                                        outputs_collections=[
                                                                                            self.satten_collections_transformer_output],
                                                                                        num_heads=self.config.num_heads,
                                                                                        seq_len_max_Q=self.token_seq_len,
                                                                                        seq_len_max_KV=self.token_seq_len)
                logging.info(
                    "[YuexiaACRLTransformerBase|transformer_layer] attn_out3 shape: {}, stt_vec_3 shape: {}, attn_mv3: {}, attn_pv3: {}".format(
                        attn_out3.get_shape().as_list(), stt_vec_3.get_shape().as_list(), attn_mv3, attn_pv3))
                attn_out3 += ff_out2
                ff_out3, ff_mv_3, ff_pv_3 = feedforward_latent_rms(inputs=attn_out3,
                                                                   cond_feat=self.latent_condition,
                                                                   num_units=[self.config.transformer_units * 4,
                                                                              self.config.transformer_units],
                                                                   activation_fn=silu,
                                                                   scope="feedforward_3",
                                                                   reuse=tf.AUTO_REUSE,
                                                                   variables_collections=[
                                                                       self.satten_collections_transformer_layer],
                                                                   outputs_collections=[
                                                                       self.satten_collections_transformer_output],
                                                                   seq_len_max=self.token_seq_len,
                                                                   hidden_dim=self.config.transformer_units)
                logging.info(
                    "[YuexiaACRLTransformerBase|transformer_layer] ff_out3 shape: {}, ff_mv_3: {}, ff_pv_3: {}".format(
                        ff_out3.get_shape().as_list(), ff_mv_3, ff_pv_3))
                ff_out3 += attn_out3

                ff_out3, _, _, _, _ = dynamic_custom_rms_norm(ff_out3, self.token_seq_len,
                                                              self.config.transformer_units, 'ffn_output_rmsnorm')

                input_shape = ff_out3.get_shape().as_list()
                ff_out3 = tf.reshape(ff_out3, [-1, input_shape[1] * input_shape[2]])
                self.transformer_output = ff_out3
                logging.info("[YuexiaACRLTransformerBase|transformer_layer] transformer_output shape: {}".format(
                    self.transformer_output.get_shape().as_list()))
        logging.info("[YuexiaACRLTransformerBase|transformer_layer] transformer_layer end: {}".format(self.transformer_output))

    def training_op(self):

        logging.info("[YX DEBUG | training_op] adagrad lr: {}, adam lr: {}, dense_adam: {}".format(
            self.training_config.initial_learning_rate, self.config.adam_lr, self.config.dense_adam))

        learning_rate_decay_cs = lambda lr, gs: learning_rate_ops.lr_cold_start(
            self.training_config.initial_learning_rate,  # 1e-5
            gs,
            self.training_config.lrcs_init_lr,  # 0.001
            self.training_config.lrcs_init_step)  # 1M
        with tf.variable_scope(name_or_scope="Optimize",
                               # partitioner=base_ops.partitioner(self.config.ps_num,
                               #                                  self.training_config.embedding_partition_size),
                               reuse=tf.AUTO_REUSE):
            if self.config.dense_adam:
                sparse_variables = []
                dense_variables = []
                for var in tf.trainable_variables():
                    if 'input_from_feature_columns' in var.name:
                        sparse_variables.append(var)
                    else:
                        dense_variables.append(var)

                logging.info('sparse_variables: {}'.format(sparse_variables))
                logging.info('dense_variables: {}'.format(dense_variables))

                train_op_vec = []
                gs = tf.train.get_or_create_global_step()
                dnn_loss1, _, _ = myopt.optimize_loss(
                    loss=self.loss,
                    global_step=self.global_step,
                    learning_rate=self.training_config.initial_learning_rate,
                    optimizer=utils.getOptimizer(self.training_config,
                                                 global_step=gs,
                                                 learning_rate=self.training_config.initial_learning_rate,
                                                 learning_rate_decay_fn=learning_rate_decay_cs),
                    update_ops=self.update_ops,
                    clip_gradients=self.training_config.clip_gradients,
                    variables=sparse_variables,
                    increment_global_step=False,
                    summaries=myopt.OPTIMIZER_SUMMARIES,
                )
                train_op_vec.append(dnn_loss1)

                dnn_loss2, _, _ = myopt.optimize_loss(
                    loss=self.loss,
                    global_step=self.global_step,
                    learning_rate=self.config.adam_lr,
                    optimizer=tf.train.AdamOptimizer(learning_rate=self.config.adam_lr, beta1=0.9, beta2=0.999,
                                                     epsilon=1e-08),
                    update_ops=self.update_ops,
                    clip_gradients=self.training_config.clip_gradients,
                    variables=dense_variables,
                    increment_global_step=False,
                    summaries=myopt.OPTIMIZER_SUMMARIES
                )
                train_op_vec.append(dnn_loss2)
                train_op_vec = control_flow_ops.group(*train_op_vec)
                with ops.control_dependencies([train_op_vec]):
                    with ops.colocate_with(self.global_step):
                        self.train_op = state_ops.assign_add(self.global_step, 1).op

            else:
                gs = tf.train.get_or_create_global_step()
                logging.info("Global_step:{},{}".format(self.name, str(gs)))

                all_trainable_vars = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
                trainable_vars = []
                logging.info("TRAINABLE_VARIABLES")
                for var in all_trainable_vars:
                    if self.is_trainable(var):
                        trainable_vars.append(var)
                        logging.info(var)

                self.train_op, _, _ = myopt.optimize_loss(
                    loss=self.loss,
                    global_step=self.global_step,
                    learning_rate=self.training_config.initial_learning_rate,
                    optimizer=utils.getOptimizer(self.training_config,
                                                 global_step=gs,
                                                 learning_rate=self.training_config.initial_learning_rate,
                                                 learning_rate_decay_fn=learning_rate_decay_cs),
                    update_ops=self.update_ops,
                    clip_gradients=self.training_config.clip_gradients,
                    variables=trainable_vars,
                    # learning_rate_decay_fn=learning_rate_decay_cs,
                    increment_global_step=True,
                    summaries=myopt.OPTIMIZER_SUMMARIES)

    def summary_op(self):
        super(YuexiaACRLTransformerBase, self).summary_op()
        sparse_params, dense_params = 0, 0
        for var in tf.trainable_variables():
            if 'input_from_feature_columns' in var.name:
                sparse_params += var.shape.num_elements()
            else:
                dense_params += var.shape.num_elements()
        tf.summary.scalar(name='{}_sparse_params_count'.format(self.name), tensor=sparse_params)
        tf.summary.scalar(name='{}_dense_params_count'.format(self.name), tensor=dense_params)