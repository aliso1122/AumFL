import torch.nn as nn
import torch
import random
from control.Enums import TrainingMode4LSTMAE

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Encoder Class
class Encoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        dropout,
        training_mode,
        encoder_only,
        
        name="",
        layer_num=1,
        encoder_only_keep_dim = True,
    ):
        super(Encoder, self).__init__()
        self.name = name
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.training_mode = training_mode
        self.encoder_only = encoder_only
        self.lstm_enc = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layer_num,
            # batch_first=True,
        )
        self.encoder_only_keep_dim = encoder_only_keep_dim

    def forward(self, x):
        self.lstm_enc.flatten_parameters()
        out, (last_h_state, last_c_state) = self.lstm_enc(
            x
        )  # last_h_state (1, batch_size, hidden_size)
        if self.training_mode in (
            TrainingMode4LSTMAE.MIXED_TEACHER_FORCING.value,
            TrainingMode4LSTMAE.RECURSIVE.value,
            TrainingMode4LSTMAE.TEACHER_FORCING.value,
        ):
            if self.encoder_only:
                return out[-1]
            return last_h_state, (last_h_state, last_c_state)
        elif self.training_mode == TrainingMode4LSTMAE.SPECIAL_UNCONDITIONED.value:
            if self.encoder_only:
                o = torch.mean(
                    out, dim=0, keepdim=self.encoder_only_keep_dim
                )  # (seq_len, batch_size, hidden_size) -> (1, batch_size, hidden_size)
                return o
            return out
        elif self.training_mode == TrainingMode4LSTMAE.UNCONDITIONED.value:
            if self.encoder_only:
                return out[-1]
            x_enc = out[-1].repeat(x.shape[0], 1, 1)
            return x_enc
        else:
            assert False

    def set_encoder_only(self):
        self.encoder_only = True

    def setback(self):
        self.encoder_only = False


# Decoder Class
class Decoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        dropout,
        use_act,
        training_mode,
        encoder_only,
        name="",
    ):
        super(Decoder, self).__init__()
        self.name = name
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_act = use_act  # Parameter to control the last sigmoid activation - depends on the normalization used.
        self.act = nn.Sigmoid()
        self.training_mode = training_mode
        self.lstm_dec = nn.LSTM(
            input_size=(
                input_size
                if self.training_mode
                in (
                    TrainingMode4LSTMAE.MIXED_TEACHER_FORCING.value,
                    TrainingMode4LSTMAE.RECURSIVE.value,
                    TrainingMode4LSTMAE.TEACHER_FORCING.value,
                )
                else hidden_size
            ),
            hidden_size=hidden_size,
            dropout=dropout,
            # batch_first=True,
        )
        self.encoder_only = encoder_only
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, enc_hiddens, enc_last_hidden_and_state=None, x=None):
        self.lstm_dec.flatten_parameters()
        dec_out = None
        if self.training_mode in (
            TrainingMode4LSTMAE.MIXED_TEACHER_FORCING.value,
            TrainingMode4LSTMAE.RECURSIVE.value,
            TrainingMode4LSTMAE.TEACHER_FORCING.value,
        ):
            teacher_forcing_ratio = 0.4
            seq_len = x.shape[0]
            batch_size = enc_last_hidden_and_state[0].shape[1]
            dec_outs = torch.zeros(seq_len, batch_size, self.input_size)

            dec_hidden = enc_last_hidden_and_state
            input = x[seq_len - 1].unsqueeze(0)
            for t in range(seq_len):
                dec_out, dec_hidden = self.lstm_dec(input, dec_hidden)
                dec_out = self.fc(dec_out)
                if self.use_act:
                    dec_out = self.act(dec_out)
                    dec_out = 2 * dec_out - 1
                dec_outs[seq_len - t - 1] = dec_out

                if (
                    self.training_mode
                    == TrainingMode4LSTMAE.MIXED_TEACHER_FORCING.value
                ):
                    if random.random() < teacher_forcing_ratio and t < seq_len - 1:
                        input = x[seq_len - t - 1 - 1].unsqueeze(0)
                    else:
                        input = dec_out
                elif self.training_mode == TrainingMode4LSTMAE.RECURSIVE.value:
                    input = dec_out
                elif self.training_mode == TrainingMode4LSTMAE.TEACHER_FORCING.value:
                    input = x[seq_len - t - 1 - 1].unsqueeze(0)
            dec_out = dec_outs

        elif self.training_mode in (
            TrainingMode4LSTMAE.SPECIAL_UNCONDITIONED.value,
            TrainingMode4LSTMAE.UNCONDITIONED.value,
        ):
            dec_out, (hidden_state, cell_state) = self.lstm_dec(enc_hiddens)
            dec_out = self.fc(dec_out)
            if self.use_act:
                dec_out = self.act(dec_out)
                dec_out = 2 * dec_out - 1
        else:
            assert False
        return dec_out

    def set_encoder_only(self):
        self.encoder_only = True

    def setback(self):
        self.encoder_only = False


# LSTM Auto-Encoder Class
class LSTMAE(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        dropout_ratio,
        training_mode,
        encoder_only,
        use_act=True,
        name="",
    ):
        super(LSTMAE, self).__init__()
        self.name = name
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio
        self.training_mode = training_mode
        self.encoder_only = encoder_only
        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout_ratio,
            training_mode=training_mode,
            encoder_only=encoder_only,
        )
        self.decoder = Decoder(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout_ratio,
            use_act=use_act,
            training_mode=training_mode,
            encoder_only=encoder_only,
        )

    def set_encoder_only(self):
        self.encoder.set_encoder_only()
        self.decoder.set_encoder_only()

    def setback(self):
        self.encoder.setback()
        self.decoder.setback()

    def forward(self, x):
        """
        x: (seq_len, batch_size, input_size)
        """
        x_dec = None
        if self.training_mode in (
            TrainingMode4LSTMAE.MIXED_TEACHER_FORCING.value,
            TrainingMode4LSTMAE.RECURSIVE.value,
            TrainingMode4LSTMAE.TEACHER_FORCING.value,
        ):
            enc_hiddens, enc_last_hidden_and_state = self.encoder(x)
            x_dec = self.decoder(
                enc_hiddens=None,
                enc_last_hidden_and_state=enc_last_hidden_and_state,
                x=x,
            )
        elif self.training_mode in (
            TrainingMode4LSTMAE.SPECIAL_UNCONDITIONED.value,
            TrainingMode4LSTMAE.UNCONDITIONED.value,
        ):
            out = self.encoder(x)
            x_dec = self.decoder(enc_hiddens=out)
        else:
            assert False

        return x_dec


class MMLSTMAE(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        dropout_ratio,
        training_mode,
        encoder_only,
        channel_num,
        input_sizes=None,
        use_act_list=None,
        name="",
    ):
        super(MMLSTMAE, self).__init__()
        self.name = name
        assert channel_num <= 3 and channel_num >= 2
        self.channel_num = channel_num
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio
        self.training_mode = training_mode
        self.encoder_only = encoder_only
        self.input_sizes = [input_size for i in range(channel_num)]
        if input_sizes is not None:
            self.input_sizes = input_sizes
        self.encoder1 = Encoder(
            input_size=self.input_sizes[0],
            hidden_size=hidden_size,
            dropout=dropout_ratio,
            training_mode=training_mode,
            encoder_only=encoder_only,
            name=f"{self.name};encoder0",
        )
        self.encoder2 = Encoder(
            input_size=self.input_sizes[1],
            hidden_size=hidden_size,
            dropout=dropout_ratio,
            training_mode=training_mode,
            encoder_only=encoder_only,
            name=f"{self.name};encoder1",
        )
        self.encoder_list = [self.encoder1, self.encoder2]

        self.decoder1 = Decoder(
            input_size=self.input_sizes[0],
            hidden_size=hidden_size,
            dropout=dropout_ratio,
            use_act=use_act_list[0],
            training_mode=training_mode,
            encoder_only=encoder_only,
            name=f"{self.name};decoder1",
        )
        self.decoder2 = Decoder(
            input_size=self.input_sizes[1],
            hidden_size=hidden_size,
            dropout=dropout_ratio,
            use_act=use_act_list[1],
            training_mode=training_mode,
            encoder_only=encoder_only,
            name=f"{self.name};decoder2",
        )
        self.decoder_list = [self.decoder1, self.decoder2]
        self.encoder3 = None
        self.decoder3 = None
        if self.channel_num == 3:
            self.encoder3 = Encoder(
                input_size=self.input_sizes[2],
                hidden_size=hidden_size,
                dropout=dropout_ratio,
                training_mode=training_mode,
                encoder_only=encoder_only,
                name=f"{self.name};encoder2",
            )
            self.decoder3 = Decoder(
                input_size=self.input_sizes[2],
                hidden_size=hidden_size,
                dropout=dropout_ratio,
                use_act=use_act_list[2],
                training_mode=training_mode,
                encoder_only=encoder_only,
                name=f"{self.name};decoder3",
            )
            self.encoder_list = [self.encoder1, self.encoder2, self.encoder3]
            self.decoder_list = [self.decoder1, self.decoder2, self.decoder3]

    def forward(self, x, channel):
        """
        x: (seq_len, batch_size, input_size)
        """
        encoder = self.encoder_list[channel]

        # decoded vectors are not less than 2 and they are stored in the list
        x_dec = []

        if self.training_mode in (
            TrainingMode4LSTMAE.MIXED_TEACHER_FORCING.value,
            TrainingMode4LSTMAE.RECURSIVE.value,
            TrainingMode4LSTMAE.TEACHER_FORCING.value,
        ):
            enc_hiddens, enc_last_hidden_and_state = encoder(x)
            # decoder channels share the same hidden vector
            for decoder in self.decoder_list:
                x_dec.append(
                    decoder(
                        enc_hiddens=None,
                        enc_last_hidden_and_state=enc_last_hidden_and_state,
                        x=x,
                    )
                )

        elif self.training_mode in (
            TrainingMode4LSTMAE.SPECIAL_UNCONDITIONED.value,
            TrainingMode4LSTMAE.UNCONDITIONED.value,
        ):
            out = encoder(x)
            for decoder in self.decoder_list:
                x_dec.append(decoder(enc_hiddens=out))
        else:
            assert False
        return x_dec

    def set_name(self, name):
        self.name = name
        for i in range(len(self.encoder_list)):
            self.encoder_list[i].name = f"{name};encoder{i}"
        for i in range(len(self.decoder_list)):
            self.decoder_list[i].name = f"{name};decoder{i}"

    def set_encoder_only(self):
        self.encoder1.set_encoder_only()
        self.decoder1.set_encoder_only()

        self.encoder2.set_encoder_only()
        self.decoder2.set_encoder_only()

        if self.encoder3 is not None:
            self.encoder3.set_encoder_only()
        if self.decoder3 is not None:
            self.decoder3.set_encoder_only()

    def setback(self):
        self.encoder1.setback()
        self.decoder1.setback()

        self.encoder2.setback()
        self.decoder2.setback()

        if self.encoder3 is not None:
            self.encoder3.setback()
        if self.decoder3 is not None:
            self.decoder3.setback()
