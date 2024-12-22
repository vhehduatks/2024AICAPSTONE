import tensorflow as tf

def get_measurement_attention_mlp(batch_size=4, num_input_features=20, list_attention_shapes=[32,16,8]) :
    x = tf.keras.Input(shape=(num_input_features,), batch_size=batch_size)
    x1 = tf.keras.layers.Dense(list_attention_shapes[0]*list_attention_shapes[0], activation='relu')(x)
    x2 = tf.keras.layers.Dense(list_attention_shapes[1]*list_attention_shapes[1], activation='relu')(x1)
    x3 = tf.keras.layers.Dense(list_attention_shapes[2]*list_attention_shapes[2], activation='relu')(x2)

    out_x1 = tf.reshape(x1, (-1,list_attention_shapes[0],list_attention_shapes[0],1))
    out_x2 = tf.reshape(x2, (-1,list_attention_shapes[1],list_attention_shapes[1],1))
    out_x3 = tf.reshape(x3, (-1,list_attention_shapes[2],list_attention_shapes[2],1))

    list_outputs = [out_x1,out_x2,out_x3]
    model = tf.keras.Model(inputs=x, outputs=list_outputs)
    return model
