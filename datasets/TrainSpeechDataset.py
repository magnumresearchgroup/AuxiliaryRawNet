import speechbrain as sb
import os


def get_dataset(hparams):
    """
    Code here is basically same with code in SpoofSpeechDataset.py
    However, audio will not be load directly.
    A random compression will be made before load by torchaudio
    """

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'spk01': 0, 'spk02': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    compress_dir = '../data/compressed_data'
    # mp3_compressor = Mp3Compression(compress_dir)

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("file_path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(file_path):
        # file_path = mp3_compressor.apply(file_path, 16000)
        # audio_id = Path(file_path).stem
        # file_path = os.path.join(
        #     compress_dir,
        #     "{}.{}".format(audio_id, choice(['mp3', 'flac']))
        # )
        sig = sb.dataio.dataio.read_audio(file_path)
        return sig

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("key")
    @sb.utils.data_pipeline.provides("key", "key_encoded")
    def label_pipeline(key):
        yield key
        key_encoded = label_encoder.encode_label_torch(key)
        yield key_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    # hparams["dataloader_options"]["shuffle"] =

    for dataset in ["train", "dev", "eval"]:
        print(hparams[f"{dataset}_annotation"])
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "key_encoded", "key"],
        )

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mapping.
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        # from_didatasets=[datasets["train"]],
        sequence_input=False,
        from_iterables=[("spoof", "bonafide")],

        # output_key="key",
    )

    return datasets