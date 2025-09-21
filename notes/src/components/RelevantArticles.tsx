function RelevantArticles({ articles }: { articles: any[] }) {
  // const articles = [
  //   {
  //     title:
  //       "Neural Machine Translation by Jointly Learning to Align and Translate",
  //     description:
  //       "This paper introduces a novel approach to neural machine translation that jointly learns to align and translate. The authors propose an attention mechanism that allows the model to focus on relevant parts of the source sentence during translation.",
  //   },
  //   {
  //     title: "Gradient-Based Learning Applied to Document Recognition",
  //     description:
  //       "This paper presents a novel approach to document recognition using gradient-based learning techniques. The authors demonstrate the effectiveness of their method on several benchmark datasets.",
  //   },
  //   {
  //     title: "Language Models are Few-Shot Learners",
  //     description:
  //       "This paper explores the capabilities of language models in few-shot learning scenarios. The authors provide empirical evidence and theoretical insights into the effectiveness of these models when faced with limited training examples.",
  //   },
  // ];

  return (
    <div className="flex flex-col text-sm w-[16rem]">
      <h2 className="font-bold">Relevant Articles</h2>
      <ul>
        {articles.map((article) => (
          <li key={article.title} className="mb-4 p-1.5">
            <h3
              className="italic text-ellipsis overflow-hidden whitespace-nowrap cursor-pointer"
              title={article.title}
            >
              {article.title}
            </h3>
            <p className="text-xs text-white/40">{article.data}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default RelevantArticles;
