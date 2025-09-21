import Left from "./components/Left";
import Right from "./components/Right";
import Bottom from "./components/Bottom";
import Top from "./components/Top";
import { useEffect, useState } from "react";
import { SpinnerCircularFixed } from "spinners-react";
// import EmbeddingAtlasWrapper from "./components/atlas/EmbeddingAtlasWrapper";

export interface Note {
  title: string;
  time: Date;
  data: {
    title: string;
    text: string;
    related: { title: string; data: string }[];
  };
}

function App() {
  const pregeneratedNotes: Note[] = [
    {
      title: "Attention Is All You Need",
      time: new Date(2025, 9, 18, 1, 30),
      data: {
        title: "Attention Is All You Need (Vaswani et al., 2017)",
        text: `This paper introduces the Transformer architecture, a sequence-to-sequence model for tasks like machine translation that relies entirely on attention mechanisms, doing away with recurrent or convolutional layers. The key idea is to use self‐attention (and cross‐attention) to allow each position in the input (or output) to attend to all other positions, thereby enabling modeling of long-range dependencies more directly and in parallel. Positional encodings are added so the model knows about sequence order. The architecture is composed of encoder and decoder stacks, each built from layers of multi‐head attention + feed-forward networks + residual connections + layer normalization. 
Empirically, the Transformer gets strong results on machine translation tasks: it outperforms previous RNN‐based or convolutional models on WMT 2014 English→German and English→French. It is faster to train because of more parallelism, and simpler in overall structure. The paper shows that with fewer sequential constraints (no RNN) and by using attention carefully (multi‐head, scaled dot-product), one can get performance plus efficiency. `,
        related: [
          {
            title:
              "“Neural Machine Translation by Jointly Learning to Align and Translate” by Bahdanau, Cho, Bengio (2014)",
            data: "foundational in introducing soft-attention in NMT. ",
          },
          {
            title:
              "“BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding” (Devlin et al., 2018) ",
            data: " builds on transformer encoders, for downstream NLP tasks.",
          },
          {
            title:
              "“Transformer-XL: Attentive Language Models Beyond a Fixed Length” (Dai et al.) ",
            data: " deals with long context modeling in Transformers.",
          },
        ],
      },
    },
    {
      title:
        "Neural Machine Translation by Jointly Learning to Align and Translate",
      time: new Date(2025, 9, 20, 4, 32),
      data: {
        title: "Attention Is All You Need (Vaswani et al., 2017)",
        text: `This paper proposes an improvement to the encoder-decoder architecture for neural machine translation by introducing soft attention. In the basic encoder-decoder model, the encoder compresses the entire source sentence into a single fixed-length vector; this becomes a bottleneck especially for long sentences. Bahdanau et al. propose letting the decoder attend over all encoder hidden states when predicting each target word, computing a context vector dynamically as a weighted sum of source hidden states (with weights (“alignments”) computed via an alignment model).
The effect is that translation quality improves, especially for longer sentences, and the model’s alignment (i.e. which source words contribute to which target words) can be inspected. They show that this soft alignment matches human intuitions in many cases. The model achieves translation performance competitive with phrase-based systems on English→French (and other) tasks, with better handling of long sentences. `,
        related: [
          {
            title:
              "“Global vs. Local Attention for Neural Machine Translation” (Luong, Pham, Manning, 2015)",
            data: "compares different forms of attention.",
          },
          {
            title:
              "“Effective Approaches to Attention-based Neural Machine Translation” (Luong, Pham, Manning, 2015) ",
            data: "various tweaks in attention and decoding.",
          },
          {
            title:
              "““Understanding the difficulty of training deep feedforward neural networks” (Glorot & Bengio, 2010)",
            data: " though not directly NMT, about problems that motivate architecture designs like attention.",
          },
        ],
      },
    },
    {
      title:
        "Listen, attend and spell: A neural network for large vocabulary conversational speech recognition",

      time: new Date(2025, 9, 21, 4, 32),
      data: {
        title: "Attention Is All You Need (Vaswani et al., 2017)",
        text: `The LAS paper proposes an end-to-end speech recognition system that transcribes raw speech (or features derived from raw speech) directly into character sequences, without relying on components like separate phoneme models, lexica, or HMM-based alignments. The model consists of two parts: Listener (encoder) and Speller (decoder). The Listener is a pyramidal recurrent neural network that transforms the acoustic signal into a compressed higher-level representation; the Speller is an attention-based decoder that predicts characters one by one, attending over the listener’s outputs. 
The LAS model allows joint learning of both acoustic modeling and language modelling (to some degree) in a single network, simplifying the pipeline. On tasks like Google Voice Search, LAS obtains reasonable word error rates (WER), especially when using techniques like beam search and optionally rescoring with language models. There are trade-offs: training is computationally expensive, alignment through attention over many acoustic frames is challenging, etc. But the results show that end-to-end methods can approach (or sometimes beat) more complex traditional speech recognition architectures.`,
        related: [
          {
            title:
              "State-of-the-art Speech Recognition With Sequence-to-Sequence Models” (Chiu, Sainath, etc., 2017)",
            data: "improves LAS with better training and architecture tweaks. ",
          },
          {
            title:
              "“Multi-Dialect Speech Recognition With A Single Sequence-To-Sequence Model” (Bo Li et al.)",
            data: " builds on transformer encoders, for downstream NLP tasks.",
          },
          {
            title:
              "“Robust Speech Recognition via Large-Scale Weak Supervision” (Whisper, Radford et al.) ",
            data: " modern scaling / supervision methods for ASR.",
          },
        ],
      },
    },
  ];

  const [bookmarked, setBookmarked] = useState<{ title: string; time: Date }[]>(
    []
  );

  // just poll the server for updating bookmarked
  useEffect(() => {
    const interval = setInterval(() => {
      fetch("http://localhost:4000/get-recent")
        .then((res) => res.json())
        .then((data) => {
          setBookmarked(
            data.history.map((i) => {
              return {
                ...i,
                title: i.url,
                time: new Date(i.timestamp),
              };
            })
          );
          console.log(data);
        });
    }, 1000); // every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const [active, setActive] = useState<number>(0);

  return (
    <>
      <div className="grid grid-cols-[14rem_auto_18rem] h-screen text-white bg-slate-800">
        <Left
          notes={pregeneratedNotes}
          bookmarked={bookmarked}
          setActive={setActive}
        />
        <div className="grid grid-rows-[min-content_auto_min-content]">
          <Top />
          <div>
            {/* <EmbeddingAtlasWrapper /> */}

            <iframe
              src="http://localhost:5055/"
              className="w-full h-full border-0"
            ></iframe>
          </div>
          <Bottom />
        </div>
        {active === -1 ? (
          <div className="flex items-center justify-center">
            <SpinnerCircularFixed color="#f54a00" />
          </div>
        ) : (
          <Right notes={pregeneratedNotes} active={active} />
        )}
      </div>
    </>
  );
}

export default App;
