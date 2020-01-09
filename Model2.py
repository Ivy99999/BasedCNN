
import tensorflow as tf
class TextCNN(object):
    def __init__(self, args):
        # Placeholders for input, output, dropout
        self.sequence_length = args.sequence_length
        self.num_classes = args.num_classes  # 类别数
        self.embedding_size = args.embedding_size
        self.filter_sizes = args.filter_sizes
        self.num_filters = args.num_filters
        self.l2_reg_lambda = args.l2_reg_lambda
        self.input_x = tf.placeholder(tf.float32, [None, self.sequence_length, self.embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.cnn()

    def cnn(self):
        with tf.device('/cpu:0'):
            embedded_chars = self.input_x
            embedded_chars_expended = tf.expand_dims(embedded_chars, -1)
            l2_loss = tf.constant(0.0)
            pooled_outputs = []
            filter_sizes = list(map(int, self.filter_sizes.split(",")))

            for i, filter_size in enumerate(filter_sizes):
                # filter_size = int(filter_size)
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution layer
                    filter_shape = [int(filter_size), self.embedding_size, 1, self.num_filters]
                    # print(filter_shape)
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        embedded_chars_expended,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.sequence_length - int(filter_size) + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="pool")
                    pooled_outputs.append(pooled)
            # Combine all the pooled features
            num_filters_total = self.num_filters * len(filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])




            # Add dropout
            with tf.name_scope("dropout"):
                h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
            # Final (unnomalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.get_variable(
                    "W",
                    shape = [num_filters_total, self.num_classes],
                    initializer = tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[self.num_classes], name = "b"))
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(h_drop, W, b, name = "scores")
                self.predictions = tf.argmax(self.scores, 1, name = "predictions")

            # Calculate Mean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.input_y)
                self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name = "accuracy")

