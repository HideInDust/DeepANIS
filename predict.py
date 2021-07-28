from keras.engine import Model
from keras.layers import Layer, Bidirectional, TimeDistributed, \
    Dense, LSTM, Masking, Input, RepeatVector, Dropout, Convolution1D, \
    BatchNormalization, Activation,Embedding,Lambda
from keras.layers.merge import concatenate, add, Multiply
import keras.backend as K
from keras.regularizers import l2

from keras_transformer.transformer import TransformerBlock
from keras_transformer.position import TransformerCoordinateEmbedding
import numpy as np
import tensorflow as tf

class MaskingByLambda(Layer):
    def __init__(self, func, **kwargs):
        self.supports_masking = True
        self.mask_func = func
        super(MaskingByLambda, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        return self.mask_func(input, input_mask)

    def call(self, x, mask=None):
        exd_mask = K.expand_dims(self.mask_func(x, mask), axis=-1)
        return x * K.cast(exd_mask, K.floatx())

def mask_by_input(tensor):
    return lambda input, mask: tensor

def base_ab_seq_model(max_cdr_len):
    input_ab = Input(shape=(max_cdr_len,))
    label_mask = Input(shape=(max_cdr_len,))
    loss_mask = Input(shape=(max_cdr_len,1))
#     pssm = Input(shape=(max_cdr_len,20))
#     hhm = Input(shape=(max_cdr_len,30))
#     spd33 = Input(shape=(max_cdr_len,14))
    transformer_depth = 2

    seq = Embedding(22, 64, input_length=max_cdr_len,mask_zero=True)(input_ab)

#     seq = concatenate([seq,pssm],axis=2)
#     seq = concatenate([seq,hhm],axis=2)
#     seq = concatenate([seq,spd33],axis=2)

    seq = MaskingByLambda(mask_by_input(label_mask))(seq)
    
    print(seq.shape)

    glb_fts = Bidirectional(LSTM(256, dropout=0.15, recurrent_dropout=0.2,
                            return_sequences=True),
                    merge_mode='concat')(seq)


    transformer_block = TransformerBlock(
        name='transformer',
        num_heads=4,
        residual_dropout=0.2,
        attention_dropout=0.2,
        use_masking=True)

    add_coordinate_embedding = TransformerCoordinateEmbedding(
        transformer_depth,
        name='coordinate_embedding')

    output = seq # shape: (<batch size>, <sequence length>, <input size>)

    output = Lambda(lambda x: x, output_shape=lambda s:s)(output)

    for step in range(transformer_depth):
        output = transformer_block(add_coordinate_embedding(output, step=step))
        
    output = MaskingByLambda(mask_by_input(label_mask))(output)
    concat_output = concatenate([glb_fts,output],axis=2)

    fts = Dropout(0.3)(concat_output)
    probs = TimeDistributed(Dense(1, activation='sigmoid',
                                  kernel_regularizer=l2(0.01)))(fts)

    probs = Multiply()([probs,loss_mask])

    return input_ab, label_mask, loss_mask, probs

def ab_seq_model(max_cdr_len):
    
    input_ab, label_mask, loss_mask, probs = base_ab_seq_model(max_cdr_len)
    print(label_mask)
    model = Model(inputs=[input_ab, label_mask,loss_mask], outputs=probs)
    print(model.summary())
    model.compile(loss= "binary_crossentropy",
                  optimizer='adam',
                  metrics=['binary_accuracy'],
                  sample_weight_mode="temporal")
    return model


def run_predict():
    
    heavy_chain = "EVQLQESGPGLVKPYQSLSLSCTVT/GYSITSDY/AWNWIRQFPGNKLEWMGYI/TYSGT/TDYNPSLKSRISITRDTSKNQFFLQLNSVTTEDTATYYCAR/YYYGYWYFDV/WGQGTTLTVSS"
    light_chain = "DIQMTQSPAIMSASPGEKVTMTC/SASSSVSYMY/WYQQKPGSSPRLLIY/DSTNLAS/GVPVRFSGSGSGTSYSLTISRMEAEDAATYYC/QQWSTYPLT/FGAGTKLELK"
    
    cdr_seq = ""
    
    cdr_seq = cdr_seq + heavy_chain.split("/")[1] + "U" + heavy_chain.split("/")[3] + "U" + heavy_chain.split("/")[5] + "U"
    cdr_seq = cdr_seq + light_chain.split("/")[1] + "U" + light_chain.split("/")[3] + "U" + light_chain.split("/")[5]

    emb_aa_s = "UCSTPAGNDEQHRKMILVFYWX"
    init_len = len(cdr_seq)
    for i in range(101-len(cdr_seq)):
        cdr_seq += 'U'


    model_factory = lambda: ab_seq_model(101)
    model = model_factory()

    cdr_seq = [emb_aa_s.index(r) for r in cdr_seq]
    cdr_seq = np.array(cdr_seq)

    masks = np.zeros(101)
    loss_masks = np.zeros(101)

    masks[0:init_len] = 1
    loss_masks[0:init_len] = 1

    masks = np.expand_dims(masks, 1)
    loss_masks = np.expand_dims(loss_masks, 1)

    cdr_seq = np.expand_dims(cdr_seq, 0)
    masks = np.expand_dims(masks, 0)
    loss_masks = np.expand_dims(loss_masks, 0)

    model.load_weights("trained_model/seq-model/seq_model.h5")

    probs_test = model.predict([cdr_seq, np.squeeze(masks,2),loss_masks])

    result_file = endpoint.get_output_file_path("predict_result.npy")

    np.save(result_file,probs_test)

    return []


if __name__ == "__main__":
    run_predict()
