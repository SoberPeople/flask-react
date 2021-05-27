import React, { useState } from "react";

function Inform({ obj }) {
  [text, setText] = useState(1);
  const Awaiting = () => setText("awating connection...");
  const connecting = () => setText("connecting to peer");

  return (
    <div>
      <h2>
        {obj}: {text}
      </h2>
    </div>
  );
}

export default Inform;
