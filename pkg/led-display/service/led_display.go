package led_display_service

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"

	log "github.com/sirupsen/logrus"

	component "github.com/nayotta/metathings/pkg/component"
)

/*
 * Kernel Configs:
 *
 * ...
 * sensor:
 *   flow_name: led_display  # flow name
 *   upload_period: 30  # seconds
 *   parser_fixed: 1
 *   data_source: "http://127.0.0.1:8080/result"
 * ...
 *
 */

type LEDDisplayService struct {
	module *component.Module
}

func (s *LEDDisplayService) get_logger() log.FieldLogger {
	return s.module.Logger()
}

func (s *LEDDisplayService) mainloop() {
	kc := s.module.Kernel().Config()

	flow := kc.GetString("sensor.flow_name")
	period := time.Duration(kc.GetInt64("sensor.upload_period")) * time.Second

	stm, err := s.module.Kernel().NewFrameStream(flow)
	if err != nil {
		s.get_logger().WithError(err).Errorf("failed to new push frame stream")
		return
	}
	defer func() {
		stm.Close()
		s.module.Stop()
		s.get_logger().Debugf("led display service mainloop exit")
	}()

	for ; ; time.Sleep(period) {
		dat, err := s.read_data()
		if err != nil {
			s.get_logger().WithError(err).Errorf("failed to read data from sensor")
			return
		}
		s.get_logger().WithField("data", dat).Debugf("read data")

		msg, err := s.parse_data(dat)
		if err != nil {
			s.get_logger().WithError(err).Warningf("failed to parse data")
			continue
		}
		s.get_logger().WithField("msg", msg).Debugf("parse data")

		err = stm.Push(msg)
		if err != nil {
			s.get_logger().WithError(err).Errorf("failed to push frame to device")
			return
		}
		s.get_logger().Debugf("push frame to flow")
	}
}

func (s *LEDDisplayService) read_data() (string, error) {
	src := s.module.Kernel().Config().GetString("sensor.data_source")
	res, err := http.Get(src)
	if err != nil {
		return "", err
	}
	defer res.Body.Close()

	buf, err := ioutil.ReadAll(res.Body)
	if err != nil {
		return "", err
	}

	msg := map[string]string{}
	err = json.Unmarshal(buf, &msg)
	if err != nil {
		return "", err
	}

	dat, _ := msg["text"]
	return dat, nil
}

func (s *LEDDisplayService) parse_data(dat string) (map[string]interface{}, error) {
	fixed := s.module.Kernel().Config().GetInt("sensor.parser_fixed")
	splited := strings.Split(dat, " ")
	if len(splited) != 2 {
		return nil, fmt.Errorf("unexpected led display data")
	}

	msg := map[string]interface{}{
		"key": splited[0],
	}

	if len(splited[1]) <= fixed {
		return nil, fmt.Errorf("unexpected led display data")
	}

	val, err := strconv.ParseFloat(splited[1], 64)
	if err != nil {
		return nil, err
	}

	val /= math.Pow(10, float64(fixed))
	msg["val"] = val

	return msg, nil
}

func (s *LEDDisplayService) InitModuleService(m *component.Module) error {
	s.module = m
	go s.mainloop()

	return nil
}
