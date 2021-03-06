import React, { Component } from "react";
import Status from "../components/Inform";
import Peer from "peerjs";
import Camera from "../components/Camera";

var lastPeerId = null;
var peer = null; // Own peer object
var peerId = null;
var hostStream = null;

// var conn = null;
var conns = {}; //접속하는 peer의 data connection
var remoteStreams = {};
var remotePeers = {};
var remoteUserCheck = {}; //guest의 눈캠, 손캠을 하나의 guestId로 묶어 저장하기 위한 배열
var remoteStreamsFlags = {};
// var recvId = document.getElementById("receiver-id");
// var status = document.getElementById("status");
var message = document.getElementById("message");
var sendMessageBox = document.getElementById("sendMessageBox");
var sendButton = document.getElementById("sendButton");
var clearMsgsButton = document.getElementById("clearMsgsButton");

/**
 * 호스트 초기화 함수
 */
function initialize() {
  // (1) Create own peer object with connection to shared PeerJS server
  const peer = new Peer(null, {
    debug: 2, //debug level:2 - print error and warnings
  });

  // 호스트 peer가 peerjs 클라우드서버에 접속되면 peer id 저장
  peer.on("open", function (id) {
    // Workaround for peer.reconnect deleting previous id
    if (peer.id === null) {
      console.log("Received null id from peer open");
      peer.id = lastPeerId; //reconnect됐을 때
    } else {
      lastPeerId = peer.id;
    }

    console.log("ID: " + peer.id);
    recvId.innerHTML = "ID: " + peer.id;
    peerId = peer.id;
    <Status text="커넥션 기다리는 중.." />;
    // status.innerHTML = "Awaiting connection...";
  });

  // 게스트 peer 접속하면 게스트별로 connection 저장
  peer.on("connection", function (c) {
    //c는 data connection
    if (conns[c.peer] && conns[c.peer].open) {
      //connection이 open되어있고, 이미 연결되어있는 peer의 connection일때: close
      // Allow only a single connection
      // if (conn && conn.open) {
      c.on("open", function () {
        c.send("Already connected to another client");
        setTimeout(function () {
          c.close();
        }, 500);
      });
      return;
    }

    c.on("data", function (data) {
      remoteUserCheck[c.peer] = data;
      console.log(
        c.peer + "의 유저의 아이디(" + remoteUserCheck[c.peer] + ") 등록"
      );
    }); // guest로 부터 전달받은 userId를 remoteUserCheck에 등록

    // conn = c;
    conns[c.peer] = c;
    console.log("Connected to: " + c.peer);
    <Status text="Connected" />;
    // status.innerHTML = "Connected";
    // ready(c.peer); //메세지ready
  });
  peer.on("disconnected", function () {
    status.innerHTML = "Connection lost. Please reconnect";
    console.log("Connection lost. Please reconnect");

    // Workaround for peer.reconnect deleting previous id
    peer.id = lastPeerId;
    peer._lastServerId = lastPeerId;
    peer.reconnect();
  });
  peer.on("close", function () {
    conn = null;
    status.innerHTML = "Connection destroyed. Please refresh";
    console.log("Connection destroyed");
  });
  peer.on("error", function (err) {
    console.log(err);
    alert("" + err);
  });

  // host화면 출력
  var getUserMedia =
    navigator.getUserMedia ||
    navigator.webkitGetUserMedia ||
    navigator.mozGetUserMedia;
  getUserMedia(
    { video: true, audio: true },
    function (stream) {
      //show local video
      addVideo(stream);

      hostStream = stream;
    },
    function (err) {
      console.log("Failed to get local stream", err);
    }
  );

  // 게스트로부터 화상전화가 오면
  peer.on("call", function (call) {
    // getUserMedia({video: true, audio: true}, function(stream) {
    var remotePeer = call.peer; //remote peer의 ID

    // 게스트에게 호스트 스트림 전달
    call.answer(hostStream); // Answer the call with an A/V stream.

    // 게스트의 스트림을 받기 위해
    call.on("stream", function (remoteStream) {
      console.log("remoteStream", remoteStream.id);
      // 이미 출력된 원격 스트림 skip
      if (remoteStreams[remoteStream.id]) {
        return;
      }
      // 각 게스트에게 새로운 게스트만 전달하도록 캐싱
      remoteStreams[remoteStream.id] = remoteStream;
      remotePeers[remoteStream.id] = remotePeer;
      // Show stream in some video/canvas element.
      console.log("remote peer", remotePeer);

      // 게스트 스트림 화면 출력
      addVideo(remoteStream);

      // 각 게스트 스트림에 나머지 모든 게스트 스트림을 전달
      for (var tmpRemoteStreamId in remoteStreams) {
        // if(tmpRemoteStreamId == remotePeer){
        //     continue;
        // }
        for (var tmpRemoteStreamId2 in remoteStreams) {
          // 같은 게스트에게 같은 스트림을 전달하지 않음.
          if (tmpRemoteStreamId == tmpRemoteStreamId2) {
            continue;
          }

          // 특정 게스트에게 이미 전달한 게스트는 통과
          if (
            remoteStreamsFlags[tmpRemoteStreamId] &&
            remoteStreamsFlags[tmpRemoteStreamId][tmpRemoteStreamId2]
          ) {
            continue;
          }

          // 특정 게스트에게 이미 전달한 게스트인지 체크하기 위한 변수 세팅
          if (!remoteStreamsFlags[tmpRemoteStreamId]) {
            remoteStreamsFlags[tmpRemoteStreamId] = {};
          }
          remoteStreamsFlags[tmpRemoteStreamId][tmpRemoteStreamId2] = true;

          console.log(
            "call remote peer",
            tmpRemoteStreamId,
            ", ",
            tmpRemoteStreamId2
          );

          // 특정 게스트에게 아직 전달하지 않은 새로운 게스트만 전달.
          peer.call(
            remotePeers[tmpRemoteStreamId],
            remoteStreams[tmpRemoteStreamId2]
          );
          // tmpCall.on('stream', function (remoteStream) {
          //     console.log("re remoteStrem");
          // });
        }
      }
    });
    // }, function(err) {
    //     console.log('Failed to get local stream' ,err);
    // });
  });
}

/**
 * 메세지 처리를 위한 세팅
 */
// function ready(client_peer_id) {
//   conns[client_peer_id].on("data", function (data) {
//     console.log("Data recieved");
//     addMessage('<span class="peerMsg">Peer: </span>' + data);

//     // 나머지 모든 게스트에게 메세진 전달.
//     for (var guest_peer in conns) {
//       if (client_peer_id == guest_peer) {
//         continue;
//       }
//       var conn = conns[guest_peer];
//       if (conn && conn.open) {
//         var msg = data;
//         sendMessageBox.value = "";
//         conn.send(msg);
//         console.log("Sent: " + msg + ", to: " + guest_peer);
//       } else {
//         console.log("Connection is closed. " + guest_peer);
//       }
//     }
//   });
//   conns[client_peer_id].on("close", function () {
//     status.innerHTML = "Connection reset<br>Awaiting connection...";
//     conn = null;
//   });
// }

// function addVideo(stream) {
//   console.log("add video");
//   // var videoTag = document.createElement("video");
//   videoTag.srcObject = stream;
//   // videoTag.muted = true;
//   // videoTag.autoplay = true;
//   document.getElementById("video_grid").append({ Camera });
// }

// function addMessage(msg) {
//   var now = new Date();
//   var h = now.getHours();
//   var m = addZero(now.getMinutes());
//   var s = addZero(now.getSeconds());

//   if (h > 12) h -= 12;
//   else if (h === 0) h = 12;

//   function addZero(t) {
//     if (t < 10) t = "0" + t;
//     return t;
//   }

//   message.innerHTML =
//     '<br><span class="msg-time">' +
//     h +
//     ":" +
//     m +
//     ":" +
//     s +
//     "</span>  -  " +
//     msg +
//     message.innerHTML;
// }

// function clearMessages() {
//   message.innerHTML = "";
//   addMessage("Msgs cleared");
// }

// // Listen for enter in message box
// sendMessageBox.addEventListener("keypress", function (e) {
//   var event = e || window.event;
//   var char = event.which || event.keyCode;
//   if (char == "13") sendButton.click();
// });
// // Send message
// sendButton.addEventListener("click", function () {
//   var msg = sendMessageBox.value;
//   sendMessageBox.value = "";
//   addMessage('<span class="selfMsg">Self: </span>' + msg);
//   for (var guest_peer in conns) {
//     var conn = conns[guest_peer];
//     if (conn && conn.open) {
//       conn.send(msg);
//       console.log("Sent: " + msg);
//     } else {
//       console.log("Connection is closed. " + guest_peer);
//     }
//   }
// });

// // Clear messages box
// clearMsgsButton.addEventListener("click", clearMessages);

export default initialize;
