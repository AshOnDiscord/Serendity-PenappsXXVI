# sen_dex-PenappsXXVI
- PennApps XXVI project made by:
- https://github.com/AshOnDiscord
- https://github.com/justanotherinternetguy
- Yuvraj Chaudhary


  ## Inspiration
The sheer amount of information accessible today is no new surprise. 

With the constant onslaught of new AI models, research papers, and academic literature, the everpresent pinnacle of society seems to be farther out of reach every second. 

How can our feeble minds comprehend, understand, and retain information to keep up with supersonic advancements in STEM? Sen_deX. And in the process, you might even get ahead. 

## Sen_deX's Purpose
Sen_deX is your subconscious second brain. 

Our background browser extension collects and filters through your search queries, screen captures, and high-attention-locations to build a representation of the concepts you learn in real time, without asking a single question. The data is then cleaned and processed into a vector embed, serving as a mathematical representation of your thoughts. A minimal notes app allows even the fleeting thoughts to be concrete through embedding. 

As you search, our extension assistant will bring the serendipitous moments to you, notifying users of concepts they have already learnt while exploring similar ones. 

But the magic happens in our search engine.

**Sen_deX is the first personalized semantic search engine to processes all of ArXiv's entire 2.6+ million paper corpus, creating the largest known personalized academic categorization engine for science research.** 

And that isn't even the end of it. 

**As the first engine to leverage inverse vector embedding in low-density regions of a given embedding space, Sen_deX predicts future concepts and research directions that haven't even yet been explored** In other words, Sen_deX _ predicts potential future research projects _.  

## How we built it
The frontend includes an app and a browser extension. The app is built with React.js and Svelte as visual frameworks and Express.js sever. We use Electron to build stable, cross-platform apps. The extension is built with Plasmo for browser integration. We use a backend of a Python Flask server to handle communication between the app, extension, and AI features. We use HuggingFace Sentence Transformers to generate embeddings and Apple's Embeddings Atlas to visualize our data. We call upon Cerebras for fast LLM inference and Exa for AI-powered recommendations and search.

## Challenges we ran into
As we progressed with Sen_deX, we pushed iitslimits farther than we had anticipated. Hosting 2.6M+ data points isn't an easy task. However, by utilizing batching and gradient accumulation, we were able to process our data in smaller chunks, making it more manageable.

## Accomplishments that we're proud of
Inverse vector embedding and latent space manipulation aren't widely studied topics, nor do they have the same traction as more accommodating options. That's why finding its potential was all the more exciting. Without adequate resources or any helpful documentation, we were able to utilize products that almost none would have touched before.

## What we learned
We learned to conduct complex data visualization and analysis on vector embeddings and latent spaces. 

## What's next for Sen_deX
Over the next few months, Sen_deX will be tested and distributed amongst professors across the nation, catered towards hyper-specific fields of study. By gathering vector data, we will be able to make inferences of the general direction of how American research moves through semantic space, accelerating our model in drawing conclusions and making predictions. Additionally, we hope Sen_deX's accessibility will be able to be utilized in the doctorate field, aiding in memory retention and documentation of dementia patients. We hope to bring Sen_deX to the investor market if possible, too.
