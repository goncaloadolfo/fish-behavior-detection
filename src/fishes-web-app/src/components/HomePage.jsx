import logo from "../images/oceanarium-logo.png";
import trackingDemo from "../videos/tracking-demo.webm";
import segmentationExample from "../videos/segmentation-example.webm";
import interpolationExample from "../videos/interpolation-example.webm";
import { useState } from "react";
import { Link } from "react-router-dom";

const HomePage = () => {
  const VIDEOS = [trackingDemo, segmentationExample, interpolationExample];

  const [currentVideo, setCurrentVideo] = useState(
    Math.floor(Math.random() * VIDEOS.length)
  );

  const updateVideo = (e) => {
    e.target.currentTime = 0;
    setCurrentVideo((currentVideo + 1) % VIDEOS.length);
  };

  return (
    <div className="container-fluid home-container">
      <div
        className="row align-items-center"
        style={{
          width: `${window.innerWidth}px`,
          height: `${window.innerHeight}px`,
        }}
      >
        <div className="col-6">
          <img src={logo} width="20%" alt="oceanarium logo"></img>

          <h1 id="homepage-title" className="display-6 fw-bold">
            Fish Behavior Detection
          </h1>

          <figure id="citation-zone">
            <blockquote className="blockquote">
              <p className="fs-4 fw-bolder">
                “A pond full of fish is better than a river full of stones.”
              </p>
            </blockquote>
            <figcaption className="blockquote-footer fs-7">
              Matshona Dhliwayo
            </figcaption>
          </figure>

          <div className="text-center">
            <button
              id="account-btn"
              type="button"
              className="btn btn-outline-primary btn-lg"
              style={{ width: "50%" }}
            >
              Ask for an account
            </button>

            <br />
            <Link to="/login">
              <button
                id="login-btn"
                type="button"
                className="btn btn-outline-secondary btn-lg"
                style={{ width: "50%" }}
              >
                Login
              </button>
            </Link>
          </div>
        </div>

        <div className="col-6 videos-col">
          {VIDEOS.map((x, index) => {
            return (
              <video
                key={index}
                style={{
                  width: `${window.innerWidth / 2}px`,
                  display: index === currentVideo ? "block" : "none",
                }}
                src={x}
                type="video/webm"
                autoPlay
                muted
                onEnded={updateVideo}
              ></video>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default HomePage;
