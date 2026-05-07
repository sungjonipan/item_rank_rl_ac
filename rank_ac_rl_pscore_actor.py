from collections import OrderedDict

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.layers.python.layers.feature_column_ops import _input_from_feature_columns
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import ops

from model_ops import checkpoint_utils
from model_ops import ops as base_ops
from model_ops import utils
from model_ops.tflog import tflogger as logging
from model_zoo.base_model_seq_actor_v1 import BaseModelSeqActorV1
from model_zoo.yuexia_ac_rl_model_pscore_v1_base import YuexiaACRLModelPScoreV1Base
from model_ops.attention import multihead_attention, feedforward
from model_ops.bert import transformer_model, transformer_model_v2


class YuexiaACRLModelPScoreV1Actor(YuexiaACRLModelPScoreV1Base):
    def __init__(self,
                 model_config,
                 training_config,
                 mc,
                 fg,
                 context,
                 name="CTR"):
        super(YuexiaACRLModelPScoreV1Actor, self).__init__(
            model_config,
            training_config,
            mc,
            fg,
            context,
            name)
        logging.info("[YuexiaACRLModelPScoreV1Actor|init] model: {}, init start.".format(self.name))

        self.context_column_blocks = ['user_columns', 'account_columns', 'context_columns']
        self.column_blocks = ['account_id_columns', 'top2_ta_query'] + self.main_column_blocks
        self.column_blocks += self.context_column_blocks

        self.actor_seq_column_blocks = ['actor_' + seq_name for seq_name in self.ta_seq_column_blocks]
        self.seq_column_blocks = ['item_score_seq', 'critic_item_score_seq'] + self.actor_seq_column_blocks + self.ta_seq_column_blocks
        self.column_blocks += [seq_name + '_length' for seq_name in self.seq_column_blocks]

        self.item_column_blocks = ['item_score_seq']
        
        self.rl_weights = self.training_config.rl_weights
        self.logp_entropy = {}
        self.print_ops = []
        
        # evaluator
        self.mtl_task = self.training_config.mtl_task

        self.critic_block_layer_dict = OrderedDict()
        self.seq_block_layer_dict = {}
        self.seq_real_len = {}

        # Define model variables collection
        self.rl_atten_collections_dnn_hidden_layer = "rl_atten_dnn_hidden_layer"
        self.rl_atten_collections_dnn_hidden_output = "rl_atten_dnn_hidden_output"
        self.rl_collections_dnn_hidden_layer = "rl_dnn_hidden_layer"
        self.rl_collections_dnn_hidden_output = "rl_dnn_hidden_output"
        self.rl_collections_layer_norm = "rl_layer_norm"
        self.rl_collections_layer_norm_output = "rl_layer_norm_output"

        self.main_net_input = None
        self.bias_net_input = None
        self.mtl_main_net = {}
        self.mtl_bias_net = {}
        self.mtl_bias = {}
        self.mtl_main_logits = {}
        self.mtl_bias_logits = {}
        self.mtl_bias = {}
        self.clkTaskName = self.training_config.clkTaskName

        logging.info("model: {}, column_blocks: {}".format(self.name, self.column_blocks))
        logging.info("model: {}, seq_column_blocks: {}".format(self.name, self.seq_column_blocks))
        logging.info("model: {}, main_column_blocks: {}".format(self.name, self.main_column_blocks))
        logging.info("model: {}, bias_column_blocks: {}".format(self.name, self.bias_column_blocks))

        # summary seq_real_lengths
        self.seq_real_lengths = OrderedDict()
        self.seq_layer_dict = OrderedDict()
        self.long_seq_columns = ['actor_long_seq']
        self.short_seq_columns = ['actor_opt_seq', 'actor_gul_seq', 'actor_ltp_seq', 'actor_realtime_click_seq',
                                  'actor_batch_pos_seq', 'actor_realtime_liveroom_seq']

        self.rl_seq_pooling_collections_dnn_hidden_layer = "rl_pooling_dnn_hidden_layer"
        self.rl_seq_pooling_collections_dnn_hidden_output = "rl_pooling_dnn_hidden_output"
        self.rl_seq_cross_att_collections_dnn_hidden_layer = "rl_cross_att_dnn_hidden_layer"
        self.rl_seq_cross_att_collections_dnn_hidden_output = "rl_cross_att_dnn_hidden_output"
        self.rl_long_seq_cross_att_collections_dnn_hidden_layer = "rl_long_seq_cross_att_dnn_hidden_layer"
        self.rl_long_seq_cross_att_collections_dnn_hidden_output = "rl_long_seq_cross_att_dnn_hidden_output"

    def build(self, features, feature_columns, labels):
        logging.info("[YuexiaACRLModelPScoreV1Actor|build] model: {}, build start.".format(self.name))

        super(YuexiaACRLModelPScoreV1Actor, self).build(features, feature_columns, labels)
        if self.training_config.model_checkpoint_dir is not None and self.training_config.model_checkpoint_dir.strip() != "":
            logging.info("loading the checkpoint: {}, {}".format(self.config.model_checkpoint_dir,
                                                                 self.config.restore_var_scope))
            checkpoint_utils.restore_from_checkpoint(ckpt_dir_or_file=self.config.model_checkpoint_dir,
                                                     restore_var_scope=self.config.restore_var_scope)

    def build_model(self):
        logging.info("[YuexiaACRLModelPScoreV1Actor|build_model] model: {}, build_model start.".format(self.name))
        if self.training_config.use_gpu:
            with tf.device('/CPU:0'):
                self.embedding_layer()
            with tf.device('/GPU:0'):
                self.actor_sequence_layer()
                self.seq_pooling_layer()
                self.model_graph()

        else:
            self.embedding_layer()
            self.actor_sequence_layer()
            self.seq_pooling_layer()
            self.model_graph()


    def model_graph(self):
        logging.info("[YuexiaACRLModelPScoreV1Actor|model_graph] model: {}, model_graph start.".format(self.name))
        with tf.variable_scope(name_or_scope="{}_rl_model".format(self.name),
                               partitioner=base_ops.partitioner(self.config.ps_num,
                                                                self.training_config.dnn_partition_size),
                               reuse=tf.AUTO_REUSE):
            self.encoder()
            self.decoder()

        # greedy
        self.rl_greedy_log_p, self.rl_greedy_pi = self.rollout('greedy')  # [N], [N, position_num]
        self.rl_reward_greedy, self.rl_pred_greedy = self.evaluate(self.rl_greedy_pi)

        # baseline
        self.bl_pi = tf.concat([self.features['rank_item1_index'], self.features['rank_item2_index']], axis=-1)
        self.bl_pi = tf.cast(self.bl_pi, tf.int32)
        self.bl_reward, self.bl_pred = self.evaluate(self.bl_pi)

        # top2_pos (0 & 1)
        pos_zero = tf.zeros_like(self.features['rank_item1_index'])
        pos_one = tf.ones_like(self.features['rank_item2_index'])

        self.top2_pi = tf.concat([pos_zero, pos_one], axis=-1)
        self.top2_pi = tf.cast(self.top2_pi, tf.int32)
        self.top2_reward, self.top2_pred = self.evaluate(self.top2_pi)
        _, self.orig_top2_pred = self.evaluate_orig()

        # sample
        if self.training_config.enable_sample_many_pi:
            rl_log_p_list, rl_pi_list = self.rollout_many_times()  # list of [N], list of [N, position_num]
            rl_reward_list = [self.evaluate(rl_pi)[0] for rl_pi in rl_pi_list]  # list of [N, 1]

            if not self.training_config.enable_sample_select_max:
                self.rl_log_p = tf.concat(rl_log_p_list, axis=0)  # [N * sample_times]
                self.rl_pi = tf.concat(rl_pi_list, axis=0)  # [N * sample_times, pos_num]
                self.rl_reward = tf.concat(rl_reward_list, axis=0)  # [N * sample_times, 1]
                bl_reward = tf.tile(self.bl_reward, [self.training_config.sample_times, 1])  # [N * sample_times, 1]
            else:  # select max reward and corresponding pi
                # add greedy results and transfer to tensor
                rl_log_p_tensor = tf.stack(rl_log_p_list + [self.rl_greedy_log_p], axis=1)  # [N, sample_times + 1]
                rl_pi_tensor = tf.stack(rl_pi_list + [self.rl_greedy_pi], axis=1)  # [N, sample_times + 1, pos_num]
                rl_reward_tensor = tf.stack(rl_reward_list + [self.rl_reward_greedy],
                                            axis=1)  # [N, sample_times + 1, 1]

                max_reward_indices = tf.squeeze(tf.cast(tf.argmax(rl_reward_tensor, axis=1), tf.int32), axis=-1)  # [N]
                self.rl_log_p = self.batch_gather(rl_log_p_tensor, max_reward_indices)  # [N]
                self.rl_pi = self.batch_gather(rl_pi_tensor, max_reward_indices)  # [N, pos_num]
                self.rl_reward = self.batch_gather(rl_reward_tensor, max_reward_indices)  # [N, 1]
                bl_reward = tf.identity(self.bl_reward)  # [N, 1]
        else:
            self.rl_log_p, self.rl_pi = self.rollout('sample')  # [N], [N, position_num]
            self.rl_reward, _ = self.evaluate(self.rl_pi)  # [N, 1]
            bl_reward = tf.identity(self.bl_reward)  # [N, 1]

        if self.config.use_cutoff_reward:
            self.rl_reward = tf.where(tf.greater_equal(self.rl_reward, 0), self.rl_reward,
                                      tf.zeros_like(self.rl_reward))

        advantage = tf.squeeze(tf.stop_gradient(self.rl_reward - bl_reward), -1)  # [N]

        if self.config.enable_normalize_advantage:
            mean = tf.reduce_mean(advantage)
            variance = tf.reduce_mean(tf.square(advantage)) - tf.square(mean)
            advantage = (advantage - mean) / tf.sqrt(variance + 1e-10)

        self.rl_loss = -tf.reduce_mean(advantage * self.rl_log_p)
        logging.info("[YuexiaACRLModelPScoreV1Actor|model_graph] model: {}, model_graph end.".format(self.name))

    def loss_op(self):
        logging.info("[YuexiaACRLModelPScoreV1Actor|loss_op] model: {}, loss_op start.".format(self.name))
        with tf.name_scope("{}_Loss_Op".format(self.name)):
            self.loss = self.rl_loss + self.reg_loss

    def encoder(self):
        logging.info("[YuexiaACRLModelPScoreV1Actor|encoder] model: {}, encoder start.".format(self.name))
        block_name = self.item_column_blocks[0]
        if not self.seq_block_layer_dict.has_key(block_name):
            logging.warn('[encoder, block layer dict] does not has block : {}'.format(block_name))
            return
        logging.info("[YuexiaACRLModelPScoreV1Actor|encoder] model: {}, encoder block_name: {}".format(self.name, block_name))

        with arg_scope(base_ops.model_arg_scope(weight_decay=self.training_config.dnn_l2_reg)):
            with tf.variable_scope(name_or_scope="encoder",
                                   partitioner=base_ops.partitioner(self.config.ps_num,
                                                                    self.training_config.dnn_partition_size),
                                   reuse=tf.AUTO_REUSE) as scope:
                
                # [N, seq_length, f_num*f_embedding]
                self.item_score_seq_feat = self.seq_block_layer_dict[block_name]
                logging.info("[YuexiaACRLModelPScoreV1Actor|encoder] model: {}, item_score_seq_feat: {}".format(self.name,  self.item_score_seq_feat))

                dim = self.config.context_embedding_hidden_units[-1]
                self.item_embeddings = self.glimpse_layer(self.item_score_seq_feat, [dim], "k")
                for seq_name in self.actor_seq_column_blocks:
                    column_name = seq_name + '_target_atten'
                    if not self.block_layer_dict.has_key(column_name):
                        logging.warn('[YuexiaACRLModelPScoreV1Actor|encoder] model: {}, block_layer_dict does not has block: {}'.format(self.name, column_name))
                        continue
                    logging.info("[YuexiaACRLModelPScoreV1Actor|encoder] model: {}, item_embeddings add feat {} : {}".format(self.name, column_name, self.block_layer_dict[column_name]))
                    self.item_embeddings = tf.add(self.item_embeddings, self.block_layer_dict[column_name])

                # self-attention
                max_len = self.fg.get_seq_len_by_sequence_name(block_name)
                self.sequence_length = self.block_layer_dict['{}_length'.format(block_name)]
                self.sequence_mask = tf.sequence_mask(tf.reshape(self.sequence_length, [-1]), max_len)   # [N, candidate_num]
                logging.info("[YuexiaACRLModelPScoreV1Actor|encoder] model: {}, sequence_mask: {}".format(self.name, self.sequence_mask))

                # [N, seq_length, 256], using transformer model
                mask = tf.cast(self.sequence_mask, tf.int32)
                mask = tf.expand_dims(mask, 1) * tf.expand_dims(mask, 2)
                self.item_embeddings = transformer_model_v2(input_tensor=self.item_embeddings,
                                                            attention_mask=mask,
                                                            num_attention_heads=self.config.num_heads,
                                                            num_hidden_layers=self.config.transformer_layers,
                                                            collections=[self.rl_atten_collections_dnn_hidden_layer],
                                                            intermediate_size=dim,
                                                            hidden_size=dim,
                                                            trainable=True,
                                                            is_training=self.is_training)
                logging.info("[YuexiaACRLModelPScoreV1Actor|encoder] model: {}, item_embeddings: {}".format(self.name, self.item_embeddings))

        logging.info("[YuexiaACRLModelPScoreV1Actor|encoder] model: {}, encoder end.".format(self.name))
    
    def glimpse_layer(self, embedding, hidden_units, name):
        for layer_id, num_hidden_units in enumerate(hidden_units):
            with variable_scope.variable_scope("glimpse_%s_layer_%d" % (name, layer_id)) as glimpse_hidden_layer_scope:
                embedding = layers.fully_connected(
                    embedding,
                    num_hidden_units,
                    activation_fn=None,
                    scope=glimpse_hidden_layer_scope,
                    variables_collections=[self.rl_collections_dnn_hidden_layer],
                    outputs_collections=[self.rl_collections_dnn_hidden_output],
                )
        return embedding

    def layer_norm(self, x, name):
        with variable_scope.variable_scope("layer_norm_%s" % (name)) as scope:
            output = layers.layer_norm(x,
                                       begin_norm_axis=-1,
                                       begin_params_axis=-1,
                                       scope=scope,
                                       variables_collections=[self.rl_collections_layer_norm],
                                       outputs_collections=[self.rl_collections_layer_norm_output])
        return output

    def decoder(self):
        logging.info("[YuexiaACRLModelPScoreV1Actor|decoder] model: {}, decoder start.".format(self.name))

        with arg_scope(base_ops.model_arg_scope(weight_decay=self.training_config.dnn_l2_reg)):
            with tf.variable_scope(name_or_scope="decoder",
                                   partitioner=base_ops.partitioner(self.config.ps_num,
                                                                    self.training_config.dnn_partition_size),
                                   reuse=tf.AUTO_REUSE) as scope:
                # [N, seq_length, 64]
                self.item_k_embedding = self.glimpse_layer(self.item_embeddings, [self.config.context_embedding_hidden_units[-1]], "k")
                self.item_v_embedding = self.glimpse_layer(self.item_embeddings, [self.config.context_embedding_hidden_units[-1]], "v")
                self.item_l_embedding = self.glimpse_layer(self.item_embeddings, [self.config.context_embedding_hidden_units[-1]], "l")   # Used to calculate output logits

                logging.info("[YuexiaACRLModelPScoreV1Actor|decoder] model: {}, item_k_embedding: {}".format(self.name, self.item_k_embedding))
                logging.info("[YuexiaACRLModelPScoreV1Actor|decoder] model: {}, item_v_embedding: {}".format(self.name, self.item_v_embedding))
                logging.info("[YuexiaACRLModelPScoreV1Actor|decoder] model: {}, item_k_embedding: {}".format(self.name, self.item_l_embedding))

                # [N, 1, 128]
                self.graph_embedding = tf.reduce_sum(self.item_embeddings, axis=1, keep_dims=True) / tf.maximum(tf.expand_dims(self.sequence_length, -1), 1.0)
                logging.info("[YuexiaACRLModelPScoreV1Actor|decoder] model: {}, graph_embedding: {}".format(self.name, self.graph_embedding))

                context_feat_list = []
                for block_name in self.context_column_blocks:
                    if not self.block_layer_dict.has_key(block_name):
                        logging.warn('[YuexiaACRLModelPScoreV1Actor|decoder] model: {}, block_layer_dict does not has block: {}'.format(self.name, block_name))
                        continue
                    context_feat_list.append(self.block_layer_dict[block_name])
                self.context_feat = tf.concat(context_feat_list, axis=-1)
                self.context_embedding = tf.concat([self.graph_embedding, tf.expand_dims(self.context_feat, axis=1)], axis=-1)
                logging.info("[YuexiaACRLModelPScoreV1Actor|decoder] model: {}, context_feat: {}".format(self.name, self.context_feat))
                logging.info("[YuexiaACRLModelPScoreV1Actor|decoder] model: {}, concat(context_embedding, context_feat): {}".format(self.name, self.context_embedding))
                
                self.context_embedding = self.glimpse_layer(self.context_embedding, self.config.context_embedding_hidden_units, "c")    # -> [N, seq_length, 256 -> 128]
                logging.info("[YuexiaACRLModelPScoreV1Actor|decoder] model: {}, context_embedding: {}".format(self.name, self.context_embedding))

                for seq_name in self.actor_seq_column_blocks:
                    column_name = seq_name + '_pooling'
                    if column_name not in self.block_layer_dict:
                        logging.warn('[YuexiaACRLModelPScoreV1Actor|decoder] model: {}, block_layer_dict does not has block: {}'.format(self.name, column_name))
                        continue
                    add_feat = tf.expand_dims(self.block_layer_dict[column_name], axis=1)
                    logging.info("[YuexiaACRLModelPScoreV1Actor|decoder] model:{}, context_embedding add feat {} : {}".format(self.name, column_name, add_feat))
                    self.context_embedding = tf.add(self.context_embedding, add_feat)
        
        logging.info('[YuexiaACRLModelPScoreV1Actor|decoder] decoder end.')

    def rollout(self, strategy):
        logging.info('[YuexiaACRLModelPScoreV1Actor|rollout] rollout start, strategy: {}'.format(strategy))
        k, v, l = self.item_k_embedding, self.item_v_embedding, self.item_l_embedding
        query = self.context_embedding
        mask = tf.logical_not(self.sequence_mask)  # [N, candidate_num]
        candidate_num = mask.get_shape().as_list()[-1]
        
        selected_item_log_p_list = []
        selected_item_index_list = []

        for i in range(self.training_config.position_num):
            log_p, _ = self.get_log_p(query, k, v, l, mask)   # [N, candidate_num]

            selected_item_index = self.select_item(log_p, strategy)  # [N]
            one_hot_mask = tf.cast(tf.one_hot(selected_item_index, candidate_num), tf.bool) # [N, candidate_num]
            mask = tf.where(one_hot_mask, tf.ones_like(mask), mask)    # update mask

            selected_item_embedding = tf.expand_dims(self.batch_gather(self.item_embeddings, selected_item_index),
                                                     axis=1)   # [N, 1, 64]
            query = query + selected_item_embedding # update query
            selected_item_log_p = self.batch_gather(log_p, selected_item_index)   # [N]

            selected_item_log_p_list.append(selected_item_log_p)
            selected_item_index_list.append(selected_item_index)

            self.logp_entropy[i] = self.get_entropy(log_p, mask)

        log_p = tf.stack(selected_item_log_p_list, axis=1)  # [N, position_num]

        log_p = tf.reduce_sum(log_p, axis=-1)               # [N]
        pi = tf.stack(selected_item_index_list, axis=1)     # [N, position_num]
        logging.info('[YuexiaACRLModelPScoreV1Actor|rollout] rollout end, strategy: {}'.format(strategy))
        return log_p, pi

    def rollout_many_times(self):
        pi_list = []
        log_p_list = []
        for _ in range(self.training_config.sample_times):
            log_p, pi = self.rollout('sample')  # [N], [N, position_num]
            log_p_list.append(log_p)
            pi_list.append(pi)
        return log_p_list, pi_list

    # q: [N, 1, dim], k & v & l: [N, candidate_num, dim]
    def get_log_p(self, q, k, v, l, mask):
        logging.info('[YuexiaACRLModelPScoreV1Actor|get_log_p] get_log_p start.')
        with tf.variable_scope(name_or_scope="{}_rl_model/rollout".format(self.name),
                               partitioner=base_ops.partitioner(self.config.ps_num,
                                                                self.training_config.dnn_partition_size),
                               reuse=tf.AUTO_REUSE) as scope:
            dim = q.get_shape().as_list()[-1]
            # cross attention
            q = self.layer_norm(q, "q")
            k = self.layer_norm(k, "k")
            compatibility = tf.matmul(q, k, transpose_b=True) / (float(dim) ** 0.5)     # [N, 1, candidate_num]
            compatibility = tf.where(mask[:, None, :], float("-inf") * tf.ones_like(compatibility), compatibility)    # [N, 1, candidate_num]
            heads = tf.matmul(tf.nn.softmax(compatibility, axis=-1), v)                 # [N, 1, dim]
            
            # output logits
            heads = self.layer_norm(heads, "heads")
            l = self.layer_norm(l, "l")
            logits = tf.matmul(heads, l, transpose_b=True) / (float(dim) ** 0.5)    # [N, 1, candidate_num]
            logits = tf.squeeze(logits, axis=1)                                     # [N, candidate_num]
            logits = tf.where(mask, float("-inf") * tf.ones_like(logits), logits)   # [N, candidate_num]
            log_p = tf.nn.log_softmax(logits / self.config.softmax_temp, axis=-1)   # [N, candidate_num]

        return log_p, heads

    # Greedy: select the node with the highest probability
    # Sample: sample according to the probability distribution
    def select_item(self, log_p, strategy):
        if strategy == 'greedy':
            selected = tf.argmax(log_p, axis=-1)
            return tf.cast(selected, tf.int32)
        else:
            selected = tf.multinomial(log_p, 1)
            return tf.squeeze(tf.cast(selected, tf.int32), axis=-1)

    def batch_gather(self, params, indices):
        # params: [N, candidate_num, dim], indices: [N]
        index = tf.range(tf.shape(params)[0])
        output_indices = tf.stack([index, indices], axis=1)  # [N, 2]
        output = tf.gather_nd(params, output_indices)  # [N, dim] or [N]
        return output
    
    def get_entropy(self, log_p, mask):
        log_p = tf.where(mask, tf.zeros_like(log_p), log_p)
        p = tf.exp(log_p)
        entropy = -tf.reduce_sum(p * log_p, axis=-1)  # [N]
        return entropy

    def setup_global_step(self):
        global_step = tf.Variable(
            initial_value=0,
            name="{}_global_step".format(self.name),
            trainable=False,
            dtype=tf.int64,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES])

        self.global_step = global_step
        self.global_step_reset = tf.assign(self.global_step, 0)
        self.global_step_add = tf.assign_add(self.global_step, 1, use_locking=True)
        tf.summary.scalar('global_step/' + self.global_step.name, self.global_step)

    def metrics_op(self):
        with tf.name_scope("{}_Metrics".format(self.name)):
            self.metrics['scalar/loss'] = self.loss
            self.metrics['scalar/loss/rl'] = self.rl_loss
            self.metrics['scalar/loss/reg'] = self.reg_loss

            self.metrics['scalar/reward/bl'] = tf.reduce_mean(self.bl_reward)
            self.metrics['scalar/reward/sample'] = tf.reduce_mean(self.rl_reward)
            self.metrics['scalar/reward/greedy'] = tf.reduce_mean(self.rl_reward_greedy)

            self.metrics['scalar/advantage/sample'] = tf.reduce_mean(self.rl_reward) - tf.reduce_mean(self.bl_reward)
            self.metrics['scalar/advantage/greedy'] = tf.reduce_mean(self.rl_reward_greedy) - tf.reduce_mean(self.bl_reward)

            self.metrics['scalar/log_p/sample'] = tf.reduce_mean(self.rl_log_p)
            self.metrics['scalar/log_p/greedy'] = tf.reduce_mean(self.rl_greedy_log_p)

            self.metrics['scalar/p/sample'] = tf.reduce_mean(tf.exp(self.rl_log_p))
            self.metrics['scalar/p/greedy'] = tf.reduce_mean(tf.exp(self.rl_greedy_log_p))

            self.metrics['scalar/entropy/step_0'] = tf.reduce_mean(self.logp_entropy[0])
            self.metrics['scalar/entropy/step_1'] = tf.reduce_mean(self.logp_entropy[1])

            for k, v in self.bl_pred.items():
                self.metrics['scalar/bl_pred/{}'.format(k)] = tf.reduce_mean(v)
            for k, v in self.rl_pred_greedy.items():
                self.metrics['scalar/rl_pred_greedy/{}'.format(k)] = tf.reduce_mean(v)
            for k, v in self.top2_pred.items():
                self.metrics['scalar/top2_pred/{}'.format(k)] = tf.reduce_mean(v)
            for k, v in self.orig_top2_pred.items():
                self.metrics['scalar/orig_top2_pred/{}'.format(k)] = tf.reduce_mean(v)

            bl_reward = tf.tile(self.bl_reward, [1, self.training_config.sample_times])
            # self.get_better_rate_and_lift(self.rl_reward, self.bl_reward, 'sample')
            # self.get_better_rate_and_lift(self.rl_reward_greedy, self.bl_reward, 'greedy')

            # # Check whether duplicate items are selected
            # self.metrics['scalar/rl_pi_unique'] = tf.reduce_mean(self.get_unique_status(self.rl_pi))
            
            logging.info("[YuexiaACRLModelPScoreV1Actor|get_log_p] metrics: {}".format(self.metrics))

    def get_unique_status(self, pi):
        def batch_unique(t):
            v, idx = tf.unique(t)
            return tf.equal(tf.shape(v)[0], self.training_config.position_num)
        unique_status = tf.map_fn(batch_unique, pi, dtype=tf.bool)
        unique_status = tf.cast(unique_status, tf.float32)
        return unique_status

    def get_better_rate_and_lift(self, rl_reward, bl_reward, strategy):
        better_index = tf.greater(rl_reward, bl_reward)
        self.metrics['scalar/better_rate/%s' % strategy] = tf.reduce_mean(
            tf.cast(better_index, tf.float32))
        worse_index = tf.greater(bl_reward, rl_reward)
        self.metrics['scalar/worse_rate/%s' % strategy] = tf.reduce_mean(
            tf.cast(worse_index, tf.float32))

        diff = rl_reward - bl_reward
        better_diff = tf.reduce_sum(tf.where(better_index, diff, tf.zeros_like(diff))) / (
            tf.reduce_sum(tf.cast(better_index, tf.float32)) + 1e-6)
        self.metrics['scalar/better_diff/%s' % strategy] = better_diff
        worse_diff = tf.reduce_sum(tf.where(worse_index, diff, tf.zeros_like(diff))) / (
            tf.reduce_sum(tf.cast(worse_index, tf.float32)) + 1e-6)
        self.metrics['scalar/worse_diff/%s' % strategy] = worse_diff
    
    def predictions_op(self):
        with tf.name_scope("{}_Predictions_Op".format(self.name)):
            self.predictions = tf.identity(self.rl_pred_greedy['is_click'])    # [N, 1]

    def mark_output(self):
        with tf.name_scope("{}_Mark_Output".format(self.name)):
            logistic = tf.identity(self.predictions, name="rank_predict")
            for taskName in self.mtl_task:
                logistic = tf.identity(self.rl_pred_greedy[taskName], name="{}_rank_predict".format(taskName))
                logistic = tf.identity(self.top2_pred[taskName], name="{}_top2_rank_predict".format(taskName))
                logistic = tf.identity(self.orig_top2_pred[taskName], name="{}_orig_top2_rank_predict".format(taskName))
            selected = tf.identity(self.rl_greedy_pi, name="item_index_rank_predict")
            reward_greedy = tf.identity(self.rl_reward_greedy, name="reward_greedy_rank_predict")
            reward_top2 = tf.identity(self.top2_reward, name="reward_top2_rank_predict")

    def summary_op(self):
        with tf.name_scope("{}_Seq_Real_len_Summary".format(self.name)):
            for block_name in self.actor_seq_column_blocks:
                seq_length = self.block_layer_dict['{}_length'.format(block_name)]
                tf.summary.scalar(name=block_name + '_mean_length', tensor=tf.reduce_mean(seq_length))

        with tf.name_scope("RL_Seq_Pooling_Summary"):
            base_ops.add_norm2_summary(self.rl_seq_pooling_collections_dnn_hidden_layer)
            base_ops.add_dense_output_summary(self.rl_seq_pooling_collections_dnn_hidden_output)
            base_ops.add_weight_summary(self.rl_seq_pooling_collections_dnn_hidden_layer)

        with tf.name_scope("RL_Seq_Atten_Summary"):
            base_ops.add_norm2_summary(self.rl_seq_cross_att_collections_dnn_hidden_layer)
            base_ops.add_dense_output_summary(self.rl_seq_cross_att_collections_dnn_hidden_output)
            base_ops.add_weight_summary(self.rl_seq_cross_att_collections_dnn_hidden_layer)

        with tf.name_scope("RL_Long_Seq_Atten_Summary"):
            base_ops.add_norm2_summary(self.rl_long_seq_cross_att_collections_dnn_hidden_layer)
            base_ops.add_dense_output_summary(self.rl_long_seq_cross_att_collections_dnn_hidden_output)
            base_ops.add_weight_summary(self.rl_long_seq_cross_att_collections_dnn_hidden_layer)

        with tf.name_scope('RL_Embedding_Summary'):
            for block_name, layer in self.block_layer_dict.items():
                if not self.mc.has_block(block_name):
                    continue
                base_ops.add_embed_layer_norm(layer, self.feature_columns[block_name], self.config.fix_sorted_columns)

        with tf.name_scope('RL_Sequence_Embedding_Summary'):
            for block_name, layer in self.seq_block_layer_dict.items():
                if not self.mc.seq_has_block(block_name):
                    continue
                seq_len = self.fg.get_seq_len_by_sequence_name(block_name)
                seq_real_length_name = '{}_length'.format(block_name)
                seq_real_length = self.block_layer_dict[seq_real_length_name]
                seq_real_length = tf.minimum(seq_real_length, tf.ones_like(seq_real_length) * seq_len)
                base_ops.add_seq_embed_layer_norm(tf.reshape(layer, [-1, tf.shape(layer)[2]]),
                                                  self.feature_columns[block_name],
                                                  seq_real_length,
                                                  self.config.fix_sorted_columns)
                logging.info('[YuexiaACRLModelPScoreV1Actor|summary_op] model: {}, seq embedding block_name: {} and columns: {}'.format(self.name, block_name, self.mc.get_seq_column_names_by_block_name(block_name)))

        with tf.name_scope("RL_Encoder_Summary"):
            base_ops.add_norm2_summary(self.rl_atten_collections_dnn_hidden_layer)
            base_ops.add_dense_output_summary(self.rl_atten_collections_dnn_hidden_output)
            base_ops.add_weight_summary(self.rl_atten_collections_dnn_hidden_layer)
        
        with tf.name_scope("RL_Decoder_Summary"):
            base_ops.add_norm2_summary(self.rl_collections_dnn_hidden_layer)
            base_ops.add_dense_output_summary(self.rl_collections_dnn_hidden_output)
            base_ops.add_weight_summary(self.rl_collections_dnn_hidden_layer)
        
        with tf.name_scope("RL_Rollout_Summary"):
            base_ops.add_weight_summary(self.rl_collections_layer_norm)
            base_ops.add_dense_output_summary(self.rl_collections_layer_norm_output)

        with tf.name_scope("RL_Metrics_Scalar"):
            for key, metric in self.metrics.items():
                tf.summary.scalar(name=key, tensor=metric)

        with tf.name_scope("Critic_Summary"):
            base_ops.add_norm2_summary(self.collections_dnn_hidden_layer)
            base_ops.add_dense_output_summary(self.collections_dnn_hidden_output)
            base_ops.add_weight_summary(self.collections_dnn_hidden_layer)

    def evaluate(self, pi):
        logging.info('[YuexiaACRLModelPScoreV1Actor|evaluate] model: {}, evaluate start.'.format(self.name))
        selected_item_emb, selected_item_ta_query = self.critic_embedding_layer(pi)
        self.update_column2emb_dict['critic_item_columns'] = selected_item_emb
        account_id_emb = self.block_layer_dict['account_id_columns']
        selected_ta_query_emb = tf.concat([account_id_emb, selected_item_ta_query], axis=-1)
        self.update_column2emb_dict['ta_query'] = selected_ta_query_emb
        logging.info('[YuexiaACRLModelPScoreV1Actor|evaluate] model: {}, selected_item_ta_query shape: {}'.format(self.name, selected_item_ta_query.get_shape().as_list()))
        logging.info('[YuexiaACRLModelPScoreV1Actor|evaluate] model: {}, selected_ta_query_emb shape: {}'.format(self.name, selected_ta_query_emb.get_shape().as_list()))

        self.seq_target_atten_layer()
        self.main_net()
        self.bias_net()
        self.logits_layer()
        mtl_pred = {}
        for task_name in self.mtl_task:
            mtl_pred[task_name] = tf.sigmoid(self.mtl_logits[task_name])

        # reward
        rewards = 0
        for key, weight in self.config.rl_weights.items():
            logging.info("[YuexiaACRLModelPScoreV1Actor|evaluate] model: {}, key: {}, weight: {}".format(self.name, key, weight))
            if key in self.config.esmm_dict:
                rewards += weight * mtl_pred[key] * mtl_pred[self.config.esmm_dict[key]]
            else:
                rewards += weight * mtl_pred[key]

        return [rewards, mtl_pred]

    def evaluate_orig(self):
        logging.info('[YuexiaACRLModelPScoreV1Actor|evaluate_orig] model: {}, evaluate_orig start.'.format(self.name))

        orig_selected_item_emb = self.block_layer_dict['critic_item_columns']
        self.update_column2emb_dict['critic_item_columns'] = orig_selected_item_emb
        top2_ta_query_emb = self.block_layer_dict['top2_ta_query']
        account_id_emb = self.block_layer_dict['account_id_columns']
        orig_selected_ta_query_emb = tf.concat([account_id_emb, top2_ta_query_emb], axis=-1)
        self.update_column2emb_dict['ta_query'] = orig_selected_ta_query_emb

        logging.info('[YuexiaACRLModelPScoreV1Actor|evaluate_orig] model: {}, orig_selected_item_emb shape: {}'.format(
            self.name, orig_selected_item_emb.get_shape().as_list()))
        logging.info('[YuexiaACRLModelPScoreV1Actor|evaluate_orig] model: {}, orig_selected_ta_query_emb shape: {}'.format(
            self.name, orig_selected_ta_query_emb.get_shape().as_list()))

        self.seq_target_atten_layer()
        self.main_net()
        self.bias_net()
        self.logits_layer()
        orig_mtl_pred = {}
        for task_name in self.mtl_task:
            orig_mtl_pred[task_name] = tf.sigmoid(self.mtl_logits[task_name])

        # reward
        rewards = 0
        for key, weight in self.config.rl_weights.items():
            logging.info("[YuexiaACRLModelPScoreV1Actor|evaluate_orig] model: {}, key: {}, weight: {}".format(self.name, key, weight))
            if key in self.config.esmm_dict:
                rewards += weight * orig_mtl_pred[key] * orig_mtl_pred[self.config.esmm_dict[key]]
            else:
                rewards += weight * orig_mtl_pred[key]

        return [rewards, orig_mtl_pred]

    def critic_embedding_layer(self, pi):
        logging.info('[YuexiaACRLModelPScoreV1Actor|critic_embedding_layer] model: {}, critic_embedding_layer start.'.format(self.name))
        with tf.variable_scope(name_or_scope="input_from_feature_columns",
                               partitioner=base_ops.partitioner(self.config.ps_num,
                                                                self.training_config.embedding_partition_size),
                               reuse=tf.AUTO_REUSE) as scope:
            seq_feature = self.seq_block_layer_dict['critic_item_score_seq']  # [B,L,N]
            feature_ = tf.batch_gather(seq_feature, pi)   # [B,L,N] -> [B,2,N]
            ta_query_feature_ = feature_
            feature = tf.reshape(feature_, [-1, feature_.get_shape().as_list()[-2]*feature_.get_shape().as_list()[-1]], name='seq_reshape')  # [B,2,N] -> [B, 2*N]
            ta_query_feature = tf.reshape(ta_query_feature_, [-1, ta_query_feature_.get_shape().as_list()[-2]*ta_query_feature_.get_shape().as_list()[-1]], name='ta_query_seq_reshape')  # [B,2,N] -> [B, 2*N]
        return feature, ta_query_feature

    def actor_sequence_layer(self):
        logging.info(
            "[BaseModelSeqActorV1|actor_sequence_layer] model: {}, actor_sequence_layer start.".format(self.name))
        with tf.variable_scope(name_or_scope="{}_rl_model".format(self.name),
                               partitioner=base_ops.partitioner(self.config.ps_num,
                                                                self.training_config.dnn_partition_size),
                               reuse=tf.AUTO_REUSE):
            self.query_name = self.item_column_blocks[0]
            if not self.seq_block_layer_dict.has_key(self.query_name):
                logging.warn('[YT DEBUG | seq] Block layer dict does not has block : {}'.format(self.query_name))
                return

            self.actor_seq_pooling_layer()
            self.actor_seq_cross_atten_layer()
            self.actor_long_seq_cross_atten_layer()

        logging.info(
            "[BaseModelSeqActorV1|actor_sequence_layer] model: {}, actor_sequence_layer end.".format(self.name))

    def actor_seq_pooling_layer(self):
        # long seq: mean -> fc; others: fc -> mean
        logging.info(
            "[BaseModelSeqActorV1|actor_seq_pooling_layer] model: {}, actor_seq_pooling_layer start.".format(self.name))

        for block_name in self.actor_seq_column_blocks:
            if not self.seq_block_layer_dict.has_key(block_name):
                logging.warn('[YT DEBUG | seq] Seq block layer dict does not has block : {}'.format(block_name))
                continue
            with arg_scope(base_ops.model_arg_scope(weight_decay=self.training_config.dnn_l2_reg)):
                with tf.variable_scope(name_or_scope=block_name + "_seq_pooling_layer",
                                       partitioner=base_ops.partitioner(self.config.ps_num,
                                                                        self.training_config.dnn_partition_size),
                                       reuse=tf.AUTO_REUSE) as scope:
                    logging.info("[YT DEBUG | seq] add {}_pooling_layer begin".format(block_name))

                    sequence = self.seq_block_layer_dict[block_name]  # [N, L, D]
                    sequence_length = self.block_layer_dict['{}_length'.format(block_name)]  # [N, 1]

                    seq_pooling = tf.reduce_sum(sequence, axis=1) / tf.maximum(sequence_length, 1.0)
                    seq_pooling = layers.fully_connected(
                        seq_pooling,
                        num_outputs=self.config.context_embedding_hidden_units[-1],
                        activation_fn=utils.getActivationFunctionOp(
                            self.config.activation_op),
                        scope=scope,
                        reuse=tf.AUTO_REUSE,
                        variables_collections=[self.rl_seq_pooling_collections_dnn_hidden_layer],
                        outputs_collections=[self.rl_seq_pooling_collections_dnn_hidden_output]
                    )

                    self.block_layer_dict[block_name + '_pooling'] = seq_pooling
                    logging.info("[YT DEBUG | seq] {}_pooling_layer: {}".format(block_name, self.block_layer_dict[
                        block_name + '_pooling']))

    def actor_seq_cross_atten_layer(self):
        logging.info(
            "[BaseModelSeqActorV1|actor_seq_cross_atten_layer] model: {}, actor_seq_cross_atten_layer start.".format(
                self.name))
        for key_name in self.short_seq_columns:
            if not self.seq_block_layer_dict.has_key(key_name):
                logging.warn(
                    '[BaseModelSeqActorV1|actor_seq_cross_atten_layer] seq_block_layer_dict dict does not has block: {}'.format(
                        self.name, key_name))
                continue
            with arg_scope(base_ops.model_arg_scope(weight_decay=self.training_config.dnn_l2_reg)):
                with tf.variable_scope(name_or_scope=key_name + "_cross_attention_layer",
                                       partitioner=base_ops.partitioner(self.config.ps_num,
                                                                        self.training_config.dnn_partition_size),
                                       reuse=tf.AUTO_REUSE) as scope:
                    query_seq = self.seq_block_layer_dict[self.query_name]  # [B, L_q, D_q]
                    query_max_len = self.fg.get_seq_len_by_sequence_name(self.query_name)
                    query_length = self.block_layer_dict['{}_length'.format(self.query_name)]
                    query_mask = tf.sequence_mask(tf.reshape(query_length, [-1]), query_max_len)  # [B, L_q]

                    key_seq = self.seq_block_layer_dict[key_name]  # [B, L_k, D_k]
                    key_max_len = self.fg.get_seq_len_by_sequence_name(key_name)
                    key_length = self.block_layer_dict['{}_length'.format(key_name)]
                    key_mask = tf.sequence_mask(tf.reshape(key_length, [-1]), key_max_len)  # [B, L_k]

                    # [B, L_q, D]
                    item_vec, _ = multihead_attention(
                        queries=query_seq,
                        keys=key_seq,
                        num_units=self.training_config.att_num_units,  # 128
                        num_output_units=self.training_config.att_num_units,  # 128
                        activation_fn=utils.getActivationFunctionOp(self.config.activation_op),  # lrelu
                        scope="cross_attention",
                        reuse=tf.AUTO_REUSE,
                        query_masks=query_mask,
                        key_masks=key_mask,
                        atten_mode=self.config.atten_mode,  # ln
                        linear_projection=self.config.sa_linear_projection,  # True
                        fix_rtp_bug=self.config.ta_fix_rtp_bug,  # False
                        variables_collections=[self.rl_seq_cross_att_collections_dnn_hidden_layer],
                        outputs_collections=[self.rl_seq_cross_att_collections_dnn_hidden_output],
                        num_heads=self.config.account_ta_num_heads,
                    )

                    # [B, L_q, D]
                    output = feedforward(
                        item_vec,
                        num_units=[self.training_config.att_num_units * 2, self.training_config.att_num_units],
                        # [64, 128]
                        activation_fn=utils.getActivationFunctionOp(self.config.activation_op),
                        scope="feed_forward",
                        reuse=tf.AUTO_REUSE,
                        variables_collections=[self.rl_seq_cross_att_collections_dnn_hidden_layer],
                        outputs_collections=[self.rl_seq_cross_att_collections_dnn_hidden_output]
                    )

                    self.block_layer_dict[key_name + '_target_atten'] = output
                    logging.info("[YT DEBUG | seq] {}_cross_attention_layer: {}".format(key_name, self.block_layer_dict[
                        key_name + '_target_atten']))

    def actor_long_seq_cross_atten_layer(self):
        logging.info(
            "[BaseModelSeqActorV1|actor_long_seq_cross_atten_layer] model: {}, actor_long_seq_cross_atten_layer start.".format(
                self.name))
        for key_name in self.long_seq_columns:
            if not self.seq_block_layer_dict.has_key(key_name):
                logging.warn(
                    '[BaseModelSeqActorV1|actor_long_seq_cross_atten_layer] model: {}, seq_block_layer_dict dict does not has block: {}'.format(
                        self.name, key_name))
                continue
            with arg_scope(base_ops.model_arg_scope(weight_decay=self.training_config.dnn_l2_reg)):
                with tf.variable_scope(name_or_scope=key_name + "_cross_attention_layer",
                                       partitioner=base_ops.partitioner(self.config.ps_num,
                                                                        self.training_config.dnn_partition_size),
                                       reuse=tf.AUTO_REUSE) as scope:
                    logging.info(
                        "[BaseModelSeqActorV1|actor_long_seq_cross_atten_layer] model: {}, add {}_cross_attention_layer begin".format(
                            self.name, key_name))
                    # key
                    key_max_len = self.fg.get_seq_len_by_sequence_name(key_name)
                    key_length = self.block_layer_dict['{}_length'.format(key_name)]
                    key_mask = tf.sequence_mask(tf.reshape(key_length, [-1]), key_max_len)  # [N, seq_length]
                    key_seq = self.seq_block_layer_dict[key_name]

                    # query
                    poly_matrix_long = tf.get_variable(
                        name="poly_anchor_{}".format(key_name),
                        shape=[self.training_config.ncodes, self.training_config.ncodes_dim],  # [32, 128]
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(128 ** (-0.5)),
                        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.MODEL_VARIABLES]
                    )
                    # metric
                    poly_norm_long = tf.nn.l2_normalize(poly_matrix_long, axis=1)  # shape=(code, fdim)
                    poly_cov_long = tf.matmul(poly_norm_long, tf.transpose(poly_norm_long))  # shape=(code, code)
                    poly_cov_long = (tf.ones_like(poly_cov_long) - tf.eye(
                        self.training_config.ncodes)) * poly_cov_long  # shape=(code, code)
                    self.cov_loss_long = tf.reduce_sum(tf.abs(poly_cov_long))

                    # Attention Query Build
                    poly_matrix_long_tile = tf.tile(tf.expand_dims(poly_matrix_long, 0),
                                                    [tf.shape(key_seq)[0], 1, 1])  # shape=(batch, code, fdim)
                    query_seq = poly_matrix_long_tile
                    query_mask = tf.sequence_mask(
                        self.training_config.ncodes * tf.ones_like(query_seq[:, 0, 0], dtype=tf.int32),
                        self.training_config.ncodes
                    )
                    sequence, stt_vec = multihead_attention(queries=query_seq,
                                                            keys=key_seq,
                                                            num_units=self.training_config.att_num_units,  # 128
                                                            num_output_units=self.training_config.att_num_units,  # 128
                                                            activation_fn=None,
                                                            scope="act_attention_1",
                                                            reuse=tf.AUTO_REUSE,
                                                            query_masks=query_mask,
                                                            key_masks=key_mask,
                                                            atten_mode=self.config.atten_mode,
                                                            linear_projection=self.config.sa_linear_projection,
                                                            fix_rtp_bug=self.config.ta_fix_rtp_bug,
                                                            variables_collections=[
                                                                self.rl_long_seq_cross_att_collections_dnn_hidden_layer],
                                                            outputs_collections=[
                                                                self.rl_long_seq_cross_att_collections_dnn_hidden_output],
                                                            num_heads=self.config.account_ta_num_heads)
                    tf.summary.histogram('[act_layer_1_stt_vec]', stt_vec)

                    atten_key = sequence  # [N, 64, 128]
                    atten_key_mask = query_mask
                    atten_query = self.seq_block_layer_dict[self.query_name]  # [B, L_q, D_q]
                    atten_query_max_len = self.fg.get_seq_len_by_sequence_name(self.query_name)
                    atten_query_length = self.block_layer_dict['{}_length'.format(self.query_name)]
                    atten_query_mask = tf.sequence_mask(tf.reshape(atten_query_length, [-1]),
                                                        atten_query_max_len)  # [B, L_q]

                    self.print_ops.append(tf.Print(atten_key, [atten_key], message="print atten_key: ", summarize=1000))
                    self.print_ops.append(
                        tf.Print(atten_key_mask, [atten_key_mask], message="print atten_key_mask: ", summarize=1000))
                    self.print_ops.append(
                        tf.Print(atten_query, [atten_query], message="print atten_query: ", summarize=1000))
                    self.print_ops.append(
                        tf.Print(atten_query_mask, [atten_query_mask], message="print atten_query_mask: ",
                                 summarize=1000))

                    # ua_item_vec: [N, L_q, 256]
                    item_vec, att_vec = multihead_attention(queries=atten_query,
                                                            keys=atten_key,
                                                            num_units=self.training_config.att_num_units,  # 128
                                                            num_output_units=self.training_config.att_num_units,  # 128
                                                            activation_fn=None,
                                                            scope="act_attention_2",
                                                            reuse=tf.AUTO_REUSE,
                                                            query_masks=atten_query_mask,
                                                            key_masks=atten_key_mask,
                                                            atten_mode=self.config.atten_mode,
                                                            linear_projection=self.config.ta_linear_projection,
                                                            fix_rtp_bug=self.config.ta_fix_rtp_bug,
                                                            variables_collections=[
                                                                self.rl_long_seq_cross_att_collections_dnn_hidden_layer],
                                                            outputs_collections=[
                                                                self.rl_long_seq_cross_att_collections_dnn_hidden_output],
                                                            num_heads=self.config.account_ta_num_heads)
                    tf.summary.histogram('act_layer_2_stt_vec', att_vec)

                    # [B, L_q, D]
                    output = feedforward(
                        item_vec,
                        num_units=[self.training_config.att_num_units * 2, self.training_config.att_num_units],
                        # [256, 128]
                        activation_fn=utils.getActivationFunctionOp(self.config.activation_op),
                        scope="feed_forward",
                        reuse=tf.AUTO_REUSE,
                        variables_collections=[self.rl_long_seq_cross_att_collections_dnn_hidden_layer],
                        outputs_collections=[self.rl_long_seq_cross_att_collections_dnn_hidden_output]
                    )

                    self.block_layer_dict[key_name + '_target_atten'] = output
                    logging.info(
                        "[BaseModelSeqActorV1|actor_long_seq_cross_atten_layer] model: {}, {}_cross_attention_layer: {}".format(
                            self.name, key_name, self.block_layer_dict[key_name + '_target_atten']))