import React from "react";

const Video = ({ className, src }) => {
  const ref = useRef < HTMLVideoElement > null;
  const videoElement = ref && ref.current;
  const classProps = classNames(styles.video, className);
  const videoSrc = src || "";

  return (
    <div>
      <video
        className={classProps}
        loop={true}
        muted={true}
        autoPlay={true}
        ref={ref}
        playsInline={true}
      >
        <source src={videoSrc} />
      </video>
    </div>
  );
};

export default Video;
