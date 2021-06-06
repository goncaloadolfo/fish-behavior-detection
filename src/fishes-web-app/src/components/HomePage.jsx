import logo from "../images/oceanarium-logo.png";
import tracking_demo from "../videos/tracking-demo.webm";
import interesting_moment from "../videos/interesting-moment-example.mp4";
import segmentation_example from "../videos/segmentation-example.webm";
import interpolation_example from "../videos/interpolation-example.webm";

function play_video(e) {
  e.preventDefault();
  let video_element = e.target;
  if (video_element.paused) video_element.play();
}

function pause_video(e) {
  e.preventDefault();
  let video_element = e.target;
  if (!video_element.paused) video_element.pause();
}

const HomePage = () => {
  return (
    <div className="container-fluid">
      <div className="row">
        <div className="col">
          <img src={logo} alt="oceanarium logo"></img>
          <h1 className="display-6">Fish Behavior Detection</h1>
          <h3>“A pond full of fish is better than a river full of stones.”</h3>
          <p>Matshona Dhliwayo</p>

          <a href="http://localhost:3000/">
            <button>Ask for an account</button>
          </a>
          <a href="http://localhost:3000/">
            <button>Login</button>
          </a>
        </div>

        <div className="col">
          <video
            width="100%"
            src={tracking_demo}
            type="video/webm"
            loop
            muted
            onMouseEnter={play_video}
            onMouseLeave={pause_video}
          ></video>

          <video
            width="100%"
            src={interesting_moment}
            type="video/mp4"
            loop
            muted
            onMouseEnter={play_video}
            onMouseLeave={pause_video}
          ></video>
        </div>

        <div className="col">
          <video
            width="100%"
            src={segmentation_example}
            type="video/webm"
            loop
            muted
            onMouseEnter={play_video}
            onMouseLeave={pause_video}
          ></video>

          <video
            width="100%"
            src={interpolation_example}
            type="video/webm"
            loop
            muted
            onMouseEnter={play_video}
            onMouseLeave={pause_video}
          ></video>
        </div>
      </div>
    </div>
  );
};

export default HomePage;
