import cPickle
import numpy as np

def load_params(model_type, dataset):
    """load the model parameters from self.model_file.
    """
    fin = open("./model/{}.{}.model.{}".format(model_type, dataset, 0))
    params = cPickle.load(fin)
    fin.close()
    return params


def store_embs(params, model_type, dataset):
    """serialize the model parameters in self.model_file.
    """
    if len(params) == 2 or len(params) == 3:
        assert params[0].shape[0] * params[0].shape[1] == \
               params[1].shape[0] * params[1].shape[1], 'shape mis-match'
        emb_all = None
        if params[0].shape == params[1].shape:
            emb_all = np.concatenate((params[0], params[1]), axis=1)
        elif params[0].shape == np.transpose(params[1]).shape:
            emb_all = np.concatenate((params[0], np.transpose(params[1])), axis=1)
        else:
            raise ValueError('Unexpected shape')

        # dump out
        print('embedding shape:', params[0].shape)
        fout = open("{}.{}.emb".format(model_type, dataset), 'w')
        cPickle.dump(params[0], fout, cPickle.HIGHEST_PROTOCOL)
        fout.close()

        print('embedding_all shape:', emb_all.shape)
        fout = open("{}.{}.emball".format(model_type, dataset), 'w')
        cPickle.dump(emb_all, fout, cPickle.HIGHEST_PROTOCOL)
        fout.close()
    else:
        raise NotImplementedError


if __name__ == '__main__':
    params = load_params('trans', 'nell.0.001')
    print(len(params))
    store_embs(params, 'trans', 'nell.0.001')