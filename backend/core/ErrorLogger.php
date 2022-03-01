<?php
namespace core;

class ErrorLogger{
    private $error_writer;
    private $exception_writer;

    function __construct(string $error_dir){
        error_reporting(\E_ALL);
        ini_set("display_errors", false);
        ini_set("log_errors", true);
        if (!is_dir($error_dir))
            mkdir($error_dir, 0777, true);
        ini_set("error_log", $error_dir . DIRECTORY_SEPARATOR . "errors.log");

        $this->error_writer = fopen($error_dir . DIRECTORY_SEPARATOR . "errors.log", "a+");
        $this->exception_writer = fopen($error_dir . DIRECTORY_SEPARATOR . "exceptions.log", "a+");

        $this->error_logger();
        $this->exception_logger();
    }

    public function log($error_no, $error_str, $error_file, $error_line){
        $this->message($error_file, $error_line, $error_str);
    }

    public function error_logger(){
        set_error_handler(function($error_no, $error_str, $error_file, $error_line){
            $this->message($error_file, $error_line, $error_str);
        });
    }

    public function exception_logger(){
        set_exception_handler(function($exception){
            $error_message = "[" . date("Y/m/d g:i:s A") . "]{\r\n";
            $error_message .= "\tError File: " . $exception->getFile() . "\r\n";
            $error_message .= "\tLine Number: " . $exception->getLine() . "\r\n";
            $error_message .= "\tError String: {\r\n";
            $error_message .= "\t    " . $exception->getMessage() . "\r\n";
            $error_message .= "\t}\r\n";
            $error_message .= "}\r\n";
            fwrite($this->exception_writer, $error_message);
        });
    }

    public function __destruct(){
        fclose($this->error_writer);
        fclose($this->exception_writer);
    }

    public function message($error_file, $error_line, $error_str): void
    {
        $error_message = "[" . date("Y/m/d g:i:s A") . "]{\r\n";
        $error_message .= "\tError File: " . $error_file . "\r\n";
        $error_message .= "\tLine Number: " . $error_line . "\r\n";
        $error_message .= "\tError String: {\r\n";
        $error_message .= "\t    " . str_replace("\n", "\n\t   \t\t", $error_str) . "\r\n";
        $error_message .= "\t}\r\n";
        $error_message .= "}\r\n";
        fwrite($this->error_writer, $error_message);
    }
}