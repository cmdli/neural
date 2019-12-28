package reddit

import (
	"io/ioutil"
	"net/http"
	"fmt"
	"encoding/json"
	"os"
	"bufio"
)

type Comment struct {
	Body string
	Score int
}

type PostData struct {
	Title string
	Ups int
}

type Post struct {
	Data PostData
}

type PostResponseData struct {
	Children []Post
}

type PostResponse struct {
	Data PostResponseData
}

func fetch(method string, url string, token string) ([]byte, error){
	client := &http.Client {
	}
	request, err := http.NewRequest(method, url, nil)
	if err != nil {
		return nil, err
	}
	request.Header.Add("Authorization", "Bearer " + token)
	request.Header.Add("User-Agent", "Post-Judge/0.0.1")

	response, err := client.Do(request)
	defer response.Body.Close()
	body, err := ioutil.ReadAll(response.Body)
	return body, err
}

func GetPosts(token string) (*PostResponse, error) {
	response, err := fetch("GET", "https://oauth.reddit.com/hot", token)
	if err != nil {
		return nil, err
	}
	fmt.Println(string(response))
	var jsonBody PostResponse
	json.Unmarshal(response, &jsonBody)
	return &jsonBody, nil
}

func ReadComments(filename string, limit int) ([]Comment, error) {
	file, err := os.Open(os.Args[1])
	if err != nil {
		return nil, err
	}
	defer file.Close()

	comments := make([]Comment, 0)
	scanner := bufio.NewScanner(file)
	scanner.Split(bufio.ScanLines)
	for scanner.Scan() && len(comments) < limit {
		var comment Comment
		json.Unmarshal(scanner.Bytes(), &comment)
		comments = append(comments, comment)
	}
	return comments, nil
}

func main() {
	response, err := GetPosts(os.Args[1])
	if err != nil {
		fmt.Println(err)
		return
	}

	file, err := os.Create(os.Args[2])
	defer file.Close()
	if err != nil {
		fmt.Println(err)
		return
	}

	data := make([]PostData, 0)
	for _, post := range response.Data.Children {
		data = append(data, post.Data)
	}

	outputJson, err := json.Marshal(data)
	if err != nil {
		fmt.Println(err)
		return
	}

	file.Write(outputJson)
}