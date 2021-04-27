import React, { Component, useRef } from "react";
import Webcam from "react-webcam";
import "./Guest.css";
// import Initialize from "./Initialize";

function Camera() {
  const camera = React.useRef();
  return (
    <div>
      <Webcam
        autoPlay
        playsInline
        muted={true}
        ref={camera}
        width="560"
        height="450"
      />
      {/* <Initialize /> */}
    </div>
  );
}

export default Camera;
