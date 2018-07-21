from __future__ import print_function

import tensorflow as tf

path_to_checkpoint = './checkpoint/bokals_1/fns.ckpt'
path_to_custom_checkpoint = './checkpoint/bokals_2/fns.ckpt'


def load_and_save(session, model_dir, target_dir):
    print("Looking for models in : " + model_dir)

    meta_file = model_dir + ".meta"

    saver = tf.train.import_meta_graph(meta_file)
    saver.restore(session, model_dir)

    print("Saving to: " + target_dir)
    saver.save(session, target_dir)


def main():
    # if not os.path.exists(path_to_custom_checkpoint):
    #     os.makedirs(path_to_custom_checkpoint)

    with tf.Graph().as_default(), tf.Session() as session:
        # test_var = tf.get_variable('test', [50, 50], dtype=tf.float32, initializer=tf.random_normal_initializer())

        # saver = tf.train.Saver()
        session.run(tf.global_variables_initializer())

        # save to the first abs path dir
        load_and_save(session, path_to_checkpoint, path_to_custom_checkpoint)


if __name__ == '__main__':
    main()
